"""
Predictive distribution interface for HomogeneousPoisson.

A homogeneous Poisson process is memoryless: the distribution of "time
until the next event" is Exponential(rate) regardless of history, so
there's no per-event state to track — every prediction uses the same
fitted rate. This is the simplest of the four predictive interfaces and a
useful reference for what the others (Hawkes, NHP, LogNormMix) generalize.
"""
from __future__ import annotations

import numpy as np

from arrivalcast.models.poisson.model import HomogeneousPoisson


class PoissonPredictive:
    def __init__(self, model: HomogeneousPoisson):
        if model.rate_per_hour is None:
            raise RuntimeError("model must be fitted first")
        self.rate = model.rate_per_hour

    def cdf(self, t: np.ndarray) -> np.ndarray:
        t = np.asarray(t, dtype=float)
        return 1.0 - np.exp(-self.rate * t)

    def survival(self, t: np.ndarray) -> np.ndarray:
        return 1.0 - self.cdf(t)

    def log_prob(self, y: float) -> float:
        """log-density of the exponential at y: log(rate) - rate*y."""
        return float(np.log(self.rate) - self.rate * y)

    def mean(self) -> float:
        return 1.0 / self.rate

    def median(self) -> float:
        return np.log(2.0) / self.rate

    def suggested_t_max(self) -> float:
        """A generous upper quadrature bound for CRPS integration — 10x the
        mean is far past where an exponential's tail is negligible."""
        return 10.0 * self.mean()
