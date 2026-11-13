"""
Predictive distribution interface for NeuralHawkes.

Unlike Hawkes, the neural intensity has no closed-form antiderivative, so
the survival function S(s) = exp(-integral_0^s lambda(u) du) must be
evaluated by numerical quadrature rather than a formula. This is still
NOT Monte Carlo sampling — evaluating lambda(u) at any single u is a cheap,
deterministic O(1) operation (CTLSTMCell.decay_to + intensity_from_hidden),
so a trapezoid-rule integral over a few hundred grid points is fast and has
no simulation noise, exactly like the CRPS quadrature in evaluation/metrics.py
(in fact this class's cdf() is typically fed straight into that function).

state (c, c_bar, o, decay_rate) must be carried forward from wherever
prediction starts — replay train, then val, then test-so-far through
NeuralHawkes.forward(), taking the state after each real event, not just
re-initialized at zero at the start of the test split. See
evaluation/evaluate.py for the replay loop.
"""
from __future__ import annotations

import numpy as np
import torch

from arrivalcast.models.neural_hawkes.model import CTLSTMCell, NeuralHawkes


class NeuralHawkesPredictive:
    """One prediction instance: distribution of the next gap given the
    CTLSTM state immediately after the preceding event."""

    def __init__(self, model: NeuralHawkes, state: tuple, n_quad_points: int = 300):
        """
        state : (c, c_bar, o, decay_rate), each a torch tensor of shape
            (hidden_dim,) — the state right after the event being
            conditioned on (i.e. what NeuralHawkes.forward() returns as
            final_state for a length-1 continuation, or the gathered
            per-sequence final_state for a longer one).
        """
        self.model = model
        self.c, self.c_bar, self.o, self.decay_rate = (t.detach() for t in state)
        self.n_quad_points = n_quad_points

    @torch.no_grad()
    def _intensity_at(self, s: np.ndarray) -> np.ndarray:
        s_t = torch.tensor(s, dtype=torch.float32).unsqueeze(-1)  # (n_s, 1)
        h = CTLSTMCell.decay_to(
            self.c.unsqueeze(0), self.c_bar.unsqueeze(0), self.decay_rate.unsqueeze(0),
            self.o.unsqueeze(0), s_t.unsqueeze(0),
        )  # (1, n_s, hidden)
        lam = self.model.intensity_from_hidden(h)  # (1, n_s)
        return lam.squeeze(0).numpy()

    @torch.no_grad()
    def survival(self, t: np.ndarray) -> np.ndarray:
        """
        Trapezoid-quadrature survival function, evaluated at every
        requested t by integrating the intensity from 0 to each t. For
        an array of t values this evaluates on a shared fine grid up to
        max(t) once, then reads off cumulative-integral values at the
        requested points — more efficient than a separate quad() call
        per t when called from crps_from_cdf's t-grid.
        """
        t = np.asarray(t, dtype=float)
        t_max = max(t.max(), 1e-6)
        grid = np.linspace(0.0, t_max, self.n_quad_points)
        lam_grid = self._intensity_at(grid)
        cumulative = np.concatenate(([0.0], np.cumsum(
            (lam_grid[1:] + lam_grid[:-1]) / 2.0 * np.diff(grid)
        )))
        compensator_at_t = np.interp(t, grid, cumulative)
        return np.exp(-compensator_at_t)

    def cdf(self, t: np.ndarray) -> np.ndarray:
        return 1.0 - self.survival(t)

    @torch.no_grad()
    def log_prob(self, y: float) -> float:
        lam_y = float(self._intensity_at(np.array([y]))[0])
        if lam_y <= 0:
            return float("-inf")
        S_y = float(self.survival(np.array([y]))[0])
        return float(np.log(lam_y) + np.log(max(S_y, 1e-300)))

    def suggested_t_max(self, fallback_hours: float = 100.0) -> float:
        """
        No closed-form mean to scale from here, so this does a coarse
        doubling search for where survival drops below 1% — cheap since
        each survival() call is itself a full quadrature, but only called
        once per prediction, not per grid point.
        """
        t = fallback_hours
        for _ in range(10):
            if self.survival(np.array([t]))[0] < 0.01:
                return t
            t *= 2.0
        return t
