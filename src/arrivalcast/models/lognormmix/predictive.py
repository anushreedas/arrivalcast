"""
Predictive distribution interface for LogNormMix.

Closed-form: the CDF of a mixture of log-normals is just the
weight-mixture of each component's log-normal CDF, which is a standard
normal CDF applied to (log t - mu_k) / sigma_k. No numerical integration
needed for the CDF itself (only quantile() needs root-finding, since the
mixture CDF has no closed-form inverse) — this is the cheapest and most
exact of the four predictive interfaces.

Note this is NOT the same as saying CRPS is closed-form for the mixture
(see evaluation/metrics.py docstring) — CRPS still goes through the same
quadrature-based crps_from_cdf() as every other model, using this exact
cdf() as its input, so LogNormMix's CRPS number is exact up to the same
quadrature grid resolution as everyone else's, not exact up to machine
precision from some separate closed-form mixture-CRPS formula.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.special import erf, logsumexp, ndtri


def _normal_cdf(z: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + erf(z / math.sqrt(2.0)))


class LogNormMixPredictive:
    """
    One prediction instance: the mixture-of-log-normals distribution of
    the next gap, given a specific decoded (log_weights, means, log_stds)
    from LogNormMixDecoder.get_params() at a particular hidden state.

    Constructed from plain numpy arrays (not torch tensors) so this class
    has no torch dependency itself — see evaluation/evaluate.py for the
    torch -> numpy handoff at the point a hidden state is decoded.
    """

    def __init__(self, weights: np.ndarray, means: np.ndarray, stds: np.ndarray):
        """
        weights : (K,) mixture weights, must sum to 1.
        means, stds : (K,) each component's log-normal parameters (i.e.
            log(gap) ~ Normal(means[k], stds[k]) within component k).
        """
        # FIX: renormalize in float64. Weights decoded from a float32 softmax
        # sum to 1 only within ~1e-7, so the mixture CDF plateaued just below 1;
        # crps_from_cdf then expanded its range looking for the tail and
        # integrated that ~1e-7 deficit out to ~1e120 hours (CRPS ~ 1e117 on
        # every source in the first preliminary run).
        w = np.clip(np.asarray(weights, dtype=float), 0.0, None)
        self.weights = w / w.sum()
        self.means = np.asarray(means, dtype=float)
        self.stds = np.asarray(stds, dtype=float)

    def cdf(self, t: np.ndarray) -> np.ndarray:
        t = np.asarray(t, dtype=float)
        log_t = np.log(np.clip(t, 1e-300, None))
        z = (log_t[:, None] - self.means[None, :]) / self.stds[None, :]
        component_cdf = _normal_cdf(z)
        return component_cdf @ self.weights

    def survival(self, t: np.ndarray) -> np.ndarray:
        return 1.0 - self.cdf(t)

    def log_prob(self, y: float) -> float:
        log_y = np.log(max(y, 1e-12))
        z = (log_y - self.means) / self.stds
        log_component = -log_y - np.log(self.stds) - 0.5 * math.log(2 * math.pi) - 0.5 * z**2
        # FIX: log-sum-exp instead of log(sum(exp(.))), which returned -inf
        # whenever every component density underflowed.
        with np.errstate(divide="ignore"):
            log_w = np.log(self.weights)
        return float(logsumexp(log_w + log_component))

    def mean(self) -> float:
        component_means = np.exp(self.means + 0.5 * self.stds**2)
        return float(np.sum(self.weights * component_means))

    def suggested_t_max(self) -> float:
        """
        A time by which the mixture's survival is ~1e-10 per significant
        component: for component k (weight w_k) the point
        exp(mu_k + z_k * sigma_k) with Phi(-z_k) = 1e-10 / w_k.

        Deliberately NOT 10 * mean(): the mean of a log-normal mixture is
        dominated by its widest component (exp(mu + sigma^2/2)) and can be
        astronomically large -- or overflow to inf -- even when the median
        is modest. This is only a starting hint for the adaptive search in
        evaluation/metrics.py.
        """
        keep = self.weights > 1e-10
        z = -ndtri(1e-10 / self.weights[keep])
        log_t = self.means[keep] + z * self.stds[keep]
        return float(np.exp(np.clip(np.max(log_t), -690.0, 700.0)))
