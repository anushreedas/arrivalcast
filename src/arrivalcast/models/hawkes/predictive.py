"""
Predictive distribution interface for HawkesExponential.

Unlike Poisson, the Hawkes predictive distribution depends on history via
the excitation state R (sum of decayed contributions from all past
events). Conveniently, for the exponential kernel this history-dependence
still has a CLOSED-FORM survival function — no Monte Carlo simulation
needed, contrary to the common assumption that intensity-based models
always require sampling for anything beyond NLL:

    lambda(s) = mu + alpha * R * exp(-beta * s)      (s = time since last event)
    S(s) = exp( -mu*s - (alpha*R/beta) * (1 - exp(-beta*s)) )

R must be carried forward from wherever prediction starts — i.e. to
correctly evaluate test-set predictions, R needs to be replayed through
train, then val, then test-so-far, not re-initialized at zero at the start
of the test split (which would silently discard the source's build-up of
self-excitation from everything before it).
"""
from __future__ import annotations

import numpy as np

from arrivalcast.models.hawkes.model import HawkesParams


def compute_R_trajectory(gaps_hours: np.ndarray, beta: float, R0: float = 0.0):
    """
    Replay a chronological gap sequence and return, for each event i, the
    R value in effect BEFORE that event (i.e. accumulated from every event
    strictly before it) — this is exactly the state HawkesPredictive needs
    to build the distribution that event i's gap is then scored against.

    Returns
    -------
    R_before : np.ndarray, shape (len(gaps_hours),), R_before[i] is the
        state used to predict gaps_hours[i]: sum_{j<=i} exp(-beta*(t_i-t_j)),
        i.e. INCLUDING event i (the arrival the forecast starts from).
    R_final : float, the state after consuming the whole sequence — pass
        this as R0 when replaying the NEXT split (e.g. val's R_final feeds
        test's R0), so state carries continuously across splits.
    """
    # FIX: the state that predicts gaps_hours[i] must include event i
    # itself. After event i the intensity is
    #     mu + alpha * sum_{j<=i} exp(-beta*(t - t_j)),
    # so R = 1 + (decayed contributions of all earlier events). The previous
    # version stored only the decayed earlier part (no "+1"), so every Hawkes
    # forecast ignored the arrival it was conditioning on. (Ozaki, 1979.)
    R = R0  # decayed excitation from events strictly before the current one
    R_before = np.empty(len(gaps_hours))
    for i, gap in enumerate(gaps_hours):
        R_before[i] = 1.0 + R
        R = np.exp(-beta * gap) * (1.0 + R)
    return R_before, R


class HawkesPredictive:
    """One prediction instance: the distribution of the NEXT gap given a
    specific excitation state R (as of just after the preceding event)."""

    def __init__(self, params: HawkesParams, R: float):
        self.mu = params.mu
        self.alpha = params.alpha
        self.beta = params.beta
        self.R = R

    def survival(self, s: np.ndarray) -> np.ndarray:
        s = np.asarray(s, dtype=float)
        compensator = self.mu * s + (self.alpha * self.R / self.beta) * (1.0 - np.exp(-self.beta * s))
        return np.exp(-compensator)

    def cdf(self, s: np.ndarray) -> np.ndarray:
        return 1.0 - self.survival(s)

    def intensity(self, s: float) -> float:
        return self.mu + self.alpha * self.R * np.exp(-self.beta * s)

    def log_prob(self, y: float) -> float:
        """log-density at y: log(lambda(y)) - compensator(y)."""
        lam = self.intensity(y)
        if lam <= 0:
            return -np.inf
        # log S(y) = -compensator(y), computed directly so a far-tail y gives a
        # large negative number instead of log(0) = -inf.
        log_S = -(self.mu * y + (self.alpha * self.R / self.beta) * (1.0 - np.exp(-self.beta * y)))
        return float(np.log(lam) + log_S)

    def suggested_t_max(self) -> float:
        """10x the (mu-only, non-excited) mean gap as a generous floor,
        widened further if the process is currently strongly excited."""
        base = 10.0 / self.mu if self.mu > 0 else 100.0
        excitation_boost = 1.0 + self.alpha * self.R / max(self.mu, 1e-6)
        return base / max(excitation_boost, 1.0) if excitation_boost > 1 else base
