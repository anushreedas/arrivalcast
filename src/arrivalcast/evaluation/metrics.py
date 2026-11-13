"""
Model-agnostic evaluation metrics.

Every function here operates purely on a predictive distribution's CDF
and/or quantile function, not on any particular model's internals. This is
the payoff of giving every model (Poisson, Hawkes, NeuralHawkes,
LogNormMix) a common interface in models/<name>/predictive.py: NLL is the
only metric computed directly from each model's own log_prob (since that's
already exact and cheap per model), while CRPS, PICP, SLA-hit, and MAE are
all written ONCE here and reused across all four.

Design choice on CRPS: rather than mixing "closed-form for LogNormMix,
Monte Carlo sampling for Hawkes/NHP" (which would make LogNormMix's number
artificially more precise than the others purely from estimator noise, not
model quality), every model exposes cdf(t) and CRPS is computed the same
way for all of them via numerical quadrature over a time grid:

    CRPS(F, y) = integral_0^inf (F(t) - 1[t >= y])^2 dt

This is exact for models with closed-form CDFs (Poisson, Hawkes, LogNormMix
all have one) and a fast, low-noise numerical approximation for NHP (whose
intensity has no closed-form antiderivative, but is O(1) to evaluate at any
single t, so a few hundred quadrature points is cheap and far more
consistent than sampling).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.optimize import brentq

_LOG_T_MIN = -690.0  # exp(-690) ~ 1e-300, smallest positive time we search
_LOG_T_MAX = 700.0   # exp(700)  ~ 1e304, largest time we search


def _cdf_at(cdf_fn: Callable[[np.ndarray], np.ndarray], t: float) -> float:
    return float(np.clip(cdf_fn(np.array([t], dtype=float))[0], 0.0, 1.0))


def _valid_hint(t_max: float | None) -> bool:
    return t_max is not None and np.isfinite(t_max) and t_max > 0


def crps_from_cdf(
    cdf_fn: Callable[[np.ndarray], np.ndarray],
    y: float,
    t_max: float | None = None,
    n_grid: int = 2000,
    tail_tol: float = 1e-9,
    rtol: float = 1e-4,
    max_expand: int = 200,
) -> float:
    """
    CRPS(F, y) = int_0^inf (F(t) - 1[t >= y])^2 dt, by adaptive quadrature.

    The integration range is found from the distribution itself rather than
    taken on trust from the caller:

    * upper bound U >= y, grown geometrically until the survival function
      S(U) <= tail_tol, so the range always covers the observation AND the
      distribution's upper tail (a too-small t_max used to truncate it);
    * lower bound L, shrunk geometrically until F(L) <= tail_tol, so mass
      near zero is resolved even when U is enormous;
    * the nodes are log-spaced on [L, y] and [y, U] separately (the
      integrand has a jump at y, which is therefore always a node), so the
      grid stays fine at the scale where F actually moves no matter how
      many orders of magnitude U spans (a linear 400-point grid out to a
      huge t_max was far too coarse there).

    The neglected pieces are bounded: int_0^L F^2 is added analytically as
    ~L*F(L)^2/3, and the tail beyond U is bounded by S(U)^2 * U, which is
    checked against `rtol` * CRPS; U is enlarged until that holds.

    t_max is accepted for API compatibility but not used: the range is
    derived from the CDF itself, so results cannot depend on a caller's
    (possibly wildly wrong) scale guess.
    """
    y = float(y)
    if y < 0:
        raise ValueError("y must be >= 0")
    trapz = getattr(np, "trapezoid", None) or np.trapz
    n_seg = max(n_grid // 2, 16)

    # Start from the observation and grow outward. t_max is deliberately not
    # used as the starting bound: an oversized hint would waste grid
    # resolution and a undersized one is corrected by the loop anyway.
    U = max(y, 1e-12)
    for _ in range(max_expand):
        if 1.0 - _cdf_at(cdf_fn, U) <= tail_tol:
            break
        U *= 4.0

    value = np.nan
    for _ in range(10):  # tail-error control loop
        # lower bound where F is negligible
        L = U if y <= 0 else min(y, U)
        for _ in range(max_expand):
            if L <= 1e-300 or _cdf_at(cdf_fn, L) <= tail_tol:
                break
            L /= 4.0

        total = 0.0
        # [0, L]: F ~ linear near 0 and F(L) <= tail_tol, so this is tiny
        total += L * _cdf_at(cdf_fn, L) ** 2 / 3.0

        # [L, y]: integrand F^2
        if y > L:
            t = np.geomspace(L, y, n_seg)
            F = np.clip(cdf_fn(t), 0.0, 1.0)
            total += float(trapz(F**2, t))
            b_start = y
        else:
            # y <= L: on [y, L] the integrand (1-F)^2 is ~1
            total += L - y
            b_start = L

        # [b_start, U]: integrand (1-F)^2
        if U > b_start:
            t = np.geomspace(b_start, U, n_seg)
            F = np.clip(cdf_fn(t), 0.0, 1.0)
            total += float(trapz((1.0 - F) ** 2, t))

        value = total
        S_U = 1.0 - _cdf_at(cdf_fn, U)
        tail_bound = S_U**2 * U
        if tail_bound <= rtol * max(value, 1e-300):
            break
        U *= 16.0
    return float(value)


def quantile_from_cdf(
    cdf_fn: Callable[[float], float],
    p: float,
    t_max: float | None = None,
    xtol: float = 1e-10,
) -> float:
    """
    Invert a monotone CDF at probability p, solving in log-time.

    The root is found for x = log t on F(exp(x)) = p. Working in log-time
    makes the problem well scaled however many orders of magnitude the
    distribution spans, and the bracket is built by expanding outward from
    the hint t_max (or t = 1) until F straddles p, so it neither needs a
    mean-based upper bound (which can be astronomically large, or inf, for a
    heavy-tailed mixture despite a sensible median) nor a hard-coded lower
    bound. xtol is in log units, i.e. a relative error on t.
    """
    if not 0 < p < 1:
        raise ValueError("p must be in (0, 1)")

    def objective(x: float) -> float:
        return _cdf_at(cdf_fn, float(np.exp(x))) - p

    x_hi = float(np.clip(np.log(t_max), _LOG_T_MIN, _LOG_T_MAX)) if _valid_hint(t_max) else 0.0
    x_lo = None
    step = 2.0
    while objective(x_hi) < 0:  # F(hi) < p: target is further out
        if x_hi >= _LOG_T_MAX:
            raise RuntimeError(f"CDF never reaches p={p} below t=exp({_LOG_T_MAX})")
        x_lo = x_hi
        x_hi = min(x_hi + step, _LOG_T_MAX)
        step *= 2.0
    if x_lo is None:
        step = 2.0
        x_lo = x_hi - step
        while objective(x_lo) > 0:  # F(lo) > p: target is further in
            if x_lo <= _LOG_T_MIN:
                return float(np.exp(_LOG_T_MIN))
            x_hi = x_lo
            x_lo = max(x_lo - step, _LOG_T_MIN)
            step *= 2.0
    root = brentq(objective, x_lo, x_hi, xtol=xtol, maxiter=500)
    return float(np.exp(root))


@dataclass
class CoverageResult:
    nominal_level: float
    n_events: int
    n_covered: int

    @property
    def observed_coverage(self) -> float:
        return self.n_covered / self.n_events if self.n_events > 0 else float("nan")


def picp(
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
    y_true: np.ndarray,
    nominal_level: float,
) -> CoverageResult:
    """
    Prediction Interval Coverage Probability: fraction of true values that
    actually fall inside their predicted [lower, upper] interval.

    nominal_level : the interval's intended coverage (e.g. 0.8 for an 80%
        interval built from quantiles 0.1 and 0.9). Passed through purely
        so the result carries its own label — this function doesn't derive
        it from the bounds.
    """
    covered = (y_true >= lower_bounds) & (y_true <= upper_bounds)
    return CoverageResult(
        nominal_level=nominal_level, n_events=len(y_true), n_covered=int(covered.sum())
    )


@dataclass
class ClassificationResult:
    tp: int
    fp: int
    fn: int
    tn: int

    @property
    def accuracy(self) -> float:
        n = self.tp + self.fp + self.fn + self.tn
        return (self.tp + self.tn) / n if n > 0 else float("nan")

    @property
    def precision(self) -> float:
        denom = self.tp + self.fp
        return self.tp / denom if denom > 0 else float("nan")

    @property
    def recall(self) -> float:
        denom = self.tp + self.fn
        return self.tp / denom if denom > 0 else float("nan")

    @property
    def f1(self) -> float:
        """F1 = 2TP / (2TP + FP + FN), defined whenever that denominator is
        positive. This equals 2PR/(P+R) when P and R both exist, but unlike
        that form it also returns 0 (not NaN) for complete failures such as
        TP=0 with FP>0 or FN>0, where precision or recall is undefined yet
        the classifier plainly scored zero. It is undefined (NaN) only when
        there are no positives at all (TP=FP=FN=0)."""
        denom = 2 * self.tp + self.fp + self.fn
        return 2 * self.tp / denom if denom > 0 else float("nan")


def sla_hit_classification(
    predicted_hit: np.ndarray,
    actual_hit: np.ndarray,
) -> ClassificationResult:
    """
    predicted_hit, actual_hit : boolean arrays, same length. "Hit" means
    "arrived within the SLA deadline." Positive class = hit.
    """
    predicted_hit = np.asarray(predicted_hit, dtype=bool)
    actual_hit = np.asarray(actual_hit, dtype=bool)
    tp = int(np.sum(predicted_hit & actual_hit))
    fp = int(np.sum(predicted_hit & ~actual_hit))
    fn = int(np.sum(~predicted_hit & actual_hit))
    tn = int(np.sum(~predicted_hit & ~actual_hit))
    return ClassificationResult(tp=tp, fp=fp, fn=fn, tn=tn)


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def pct_improvement(baseline_value: float, model_value: float) -> float:
    """
    % reduction in an error metric (MAE/RMSE/NLL — lower-is-better) versus
    baseline. Positive = model is better than baseline.
    """
    if baseline_value == 0:
        return float("nan")
    return 100.0 * (baseline_value - model_value) / abs(baseline_value)
