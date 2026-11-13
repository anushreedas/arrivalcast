"""
Univariate Hawkes process with a single exponential decay kernel:

    lambda(t) = mu + alpha * sum_{t_i < t} exp(-beta * (t - t_i))

Fit by maximum likelihood on the raw event times of one source. Like the
Poisson baseline, this classical form has no slot for calendar covariates
or Time2Vec embeddings — mu, alpha, and beta are three scalars per source,
not learned functions of side information. If you want calendar structure
in a Hawkes-family model, the standard extension is a time-varying baseline
mu(t) (e.g. mu estimated per calendar bucket, analogous to
PiecewiseConstantPoisson); that's a meaningfully different model from the
one implemented here, not a feature you bolt on to this class.

Log-likelihood uses the standard recursive trick (Ozaki, 1979) to compute
the sum-of-exponentials term in O(n) rather than O(n^2).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize


@dataclass
class HawkesParams:
    mu: float
    alpha: float
    beta: float

    @property
    def branching_ratio(self) -> float:
        """alpha/beta = expected number of 'child' events per 'parent'
        event. Must be < 1 for the process to be stationary (non-exploding)."""
        return self.alpha / self.beta if self.beta > 0 else np.inf


class HawkesExponential:
    def __init__(self):
        self.params: HawkesParams | None = None

    @staticmethod
    def _neg_log_likelihood(theta: np.ndarray, gaps_hours: np.ndarray, total_hours: float) -> float:
        mu, alpha, beta = theta
        if mu <= 0 or alpha < 0 or beta <= 0:
            return np.inf

        # Recursive computation of R_i = sum_{t_j < t_i} exp(-beta*(t_i - t_j))
        # gaps_hours[i] = t_{i+1} - t_i, so R_0 = 0 and
        # R_i = exp(-beta*gap_{i-1}) * (1 + R_{i-1})
        n = len(gaps_hours) + 1
        R = 0.0
        sum_log_intensity = np.log(mu)  # term for the first event (R_0 = 0)
        for gap in gaps_hours:
            R = np.exp(-beta * gap) * (1.0 + R)
            intensity = mu + alpha * R
            if intensity <= 0:
                return np.inf
            sum_log_intensity += np.log(intensity)

        # Compensator term: integral of lambda(t) dt over [0, T]
        # = mu*T + (alpha/beta) * sum_i (1 - exp(-beta*(T - t_i)))
        compensator = mu * total_hours
        compensator += (alpha / beta) * (n - _sum_exp_tail(gaps_hours, beta, total_hours))

        nll = -(sum_log_intensity - compensator)
        return nll

    def fit(
        self,
        gaps_hours: np.ndarray,
        total_hours: float,
        inits: list[tuple[float, float, float]] | None = None,
    ) -> HawkesExponential:
        """
        gaps_hours : inter-arrival gaps for the training split, in hours.
        total_hours : length of the training observation window, in hours
            (used for the compensator / survival term).
        inits : list of (mu, alpha, beta) starting points. Nelder-Mead on
            this likelihood surface can converge to a spurious "no
            self-excitation" local optimum (alpha/beta collapsing towards
            0) depending on the start point — empirically this happens for
            roughly half of naive single-point starts on this project's
            synthetic Hawkes source. We therefore multi-start from several
            points spanning weak/strong and fast/slow decay and keep
            whichever converges to the highest log-likelihood. Callers who
            know a good starting point (e.g. from a previous fit) can pass
            a single-element list to skip the extra restarts.
        """
        if inits is None:
            inits = [
                (0.1, 0.5, 1.0),
                (0.05, 0.3, 0.5),
                (0.05, 0.8, 2.0),
                (0.02, 1.0, 0.25),
                (0.1, 0.1, 4.0),
            ]

        best_result, best_nll = None, np.inf
        for init in inits:
            result = minimize(
                self._neg_log_likelihood,
                x0=np.array(init, dtype=float),
                args=(gaps_hours, total_hours),
                method="Nelder-Mead",
                options={"xatol": 1e-8, "fatol": 1e-8, "maxiter": 5000},
            )
            if result.fun < best_nll:
                best_nll, best_result = result.fun, result

        mu, alpha, beta = best_result.x
        self.params = HawkesParams(mu=mu, alpha=alpha, beta=beta)
        self._fit_result = best_result
        return self

    def log_likelihood(self, gaps_hours: np.ndarray, total_hours: float) -> float:
        if self.params is None:
            raise RuntimeError("call .fit() first")
        theta = (self.params.mu, self.params.alpha, self.params.beta)
        return -self._neg_log_likelihood(np.array(theta), gaps_hours, total_hours)

    def intensity_at_end(self, gaps_hours: np.ndarray) -> float:
        """Current intensity lambda(t) immediately after the last event —
        useful as a quick "how excited is the process right now" diagnostic."""
        if self.params is None:
            raise RuntimeError("call .fit() first")
        mu, alpha, beta = self.params.mu, self.params.alpha, self.params.beta
        R = 0.0
        for gap in gaps_hours:
            R = np.exp(-beta * gap) * (1.0 + R)
        return mu + alpha * R


def _sum_exp_tail(gaps_hours: np.ndarray, beta: float, total_hours: float) -> float:
    """
    Helper for the compensator term: sum_i exp(-beta*(T - t_i)) computed via
    a forward recursion analogous to R_i, reusing cumulative elapsed time.
    Returned so the caller can compute n - this to get sum_i (1 - exp(...)).
    """
    elapsed = np.concatenate(([0.0], np.cumsum(gaps_hours)))
    remaining = total_hours - elapsed
    remaining = np.clip(remaining, 0.0, None)
    return float(np.sum(np.exp(-beta * remaining)))
