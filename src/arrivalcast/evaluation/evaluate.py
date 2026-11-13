"""
Turns a list of per-test-event predictions into one source's metric row.

This is the one place all six evaluation ideas (NLL, CRPS, PICP, SLA-hit,
MAE, baseline comparison) come together, and it's written once against a
minimal, model-agnostic prediction record (PerEventPrediction) rather than
against any particular model's internals — see models/<name>/evaluate.py
for the per-model code that builds these records (replaying history state
correctly, fitting, etc.) and calls aggregate_source_metrics() at the end.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from arrivalcast.evaluation.metrics import (
    ClassificationResult,
    crps_from_cdf,
    mae,
    picp,
    quantile_from_cdf,
    rmse,
    sla_hit_classification,
)

DEFAULT_NOMINAL_LEVELS = (0.5, 0.8, 0.9)


@dataclass
class PerEventPrediction:
    """One test-set event's true outcome plus its predictive distribution,
    expressed only through cdf() (and log_prob(), which every model
    computes directly and cheaply rather than deriving from cdf)."""

    y_true: float
    cdf_fn: Callable[[np.ndarray], np.ndarray]
    log_prob: float
    t_max: float  # starting hint for the adaptive CRPS/quantile search (see
    # evaluation/metrics.py); results don't depend on it being tight


@dataclass
class SourceMetrics:
    source_id: str
    model: str
    n_test_events: int
    low_n: bool

    mean_nll: float
    mean_crps: float
    mae: float
    rmse: float

    coverage: dict  # nominal_level -> CoverageResult
    sla: ClassificationResult
    sla_threshold_hours: float

    baseline_mae: float | None = None
    mae_pct_improvement: float | None = None

    # One row per test event (see aggregate_source_metrics); None only if the
    # caller did not ask for it. Not included in source_metrics_to_row().
    per_event: pd.DataFrame | None = None


def aggregate_source_metrics(
    source_id: str,
    model_name: str,
    predictions: list[PerEventPrediction],
    sla_hours: float,
    low_n_threshold: int = 5,
    nominal_levels: tuple = DEFAULT_NOMINAL_LEVELS,
    baseline_predictions: np.ndarray | None = None,
    crps_n_grid: int = 2000,
    timestamps: pd.Series | None = None,
    event_positions: np.ndarray | None = None,
) -> SourceMetrics:
    """
    predictions : one PerEventPrediction per test event, in chronological
        order.
    timestamps, event_positions : optional. The source's full sorted arrival
        timestamps and the positions (in that series) of the test events, one
        per prediction. When given, the per-event table in SourceMetrics.per_event
        also carries each event's actual arrival time and the time it was
        predicted from (the previous arrival).
    baseline_predictions : optional, same length as predictions — the
        naive lag-1 point forecast for each test event (see
        evaluation/baseline.py). If provided, baseline_mae and
        mae_pct_improvement are filled in.
    """
    n = len(predictions)
    if n == 0:
        raise ValueError(f"no test predictions for {source_id}/{model_name}")

    y_true = np.array([p.y_true for p in predictions])

    # NLL: mean of -log_prob, already exact/cheap per model.
    mean_nll = -np.mean([p.log_prob for p in predictions])

    # CRPS: uniform quadrature-based estimator for every model (see
    # evaluation/metrics.py docstring on why this isn't mixed with a
    # closed-form shortcut for any single model).
    crps_values = np.array(
        [crps_from_cdf(p.cdf_fn, p.y_true, p.t_max, n_grid=crps_n_grid) for p in predictions]
    )
    mean_crps = float(np.mean(crps_values))

    # Point forecast for MAE/RMSE: the predictive median, from the same
    # cdf() every other metric uses (no separate "mean_gap" formula
    # needed per model, keeping this generic).
    medians = np.array([quantile_from_cdf(p.cdf_fn, 0.5, p.t_max) for p in predictions])
    quantile_cols = {0.5: medians}  # kept for the per-event table (no recomputation)
    mae_val = mae(y_true, medians)
    rmse_val = rmse(y_true, medians)

    # PICP at each requested nominal level, symmetric two-sided interval
    # (e.g. 80% level -> [quantile(0.1), quantile(0.9)]).
    coverage = {}
    for level in nominal_levels:
        lo_p, hi_p = (1 - level) / 2.0, 1 - (1 - level) / 2.0
        lowers = np.array([quantile_from_cdf(p.cdf_fn, lo_p, p.t_max) for p in predictions])
        uppers = np.array([quantile_from_cdf(p.cdf_fn, hi_p, p.t_max) for p in predictions])
        coverage[level] = picp(lowers, uppers, y_true, nominal_level=level)
        quantile_cols[lo_p], quantile_cols[hi_p] = lowers, uppers

    # SLA-hit: predicted "hit" if the model assigns >= 50% probability
    # mass to arriving within sla_hours (equivalently: predicted median
    # <= sla_hours). Actual "hit" if the true gap is within sla_hours.
    predicted_hit = medians <= sla_hours
    actual_hit = y_true <= sla_hours
    sla_result = sla_hit_classification(predicted_hit, actual_hit)

    baseline_mae_val = None
    pct_improve = None
    if baseline_predictions is not None:
        valid = ~np.isnan(baseline_predictions)
        if valid.any():
            baseline_mae_val = mae(y_true[valid], np.asarray(baseline_predictions)[valid])
            model_mae_on_same_subset = mae(y_true[valid], medians[valid])
            pct_improve = 100.0 * (baseline_mae_val - model_mae_on_same_subset) / abs(baseline_mae_val) \
                if baseline_mae_val != 0 else float("nan")

    # ---- per-event table: the same numbers the means above were taken over ----
    pit = np.array([float(np.asarray(p.cdf_fn(np.array([p.y_true])))[0]) for p in predictions])
    per_event = pd.DataFrame({
        "y_true": y_true,                                     # actual gap, hours
        "log_prob": [p.log_prob for p in predictions],
        "nll": [-p.log_prob for p in predictions],
        "crps": crps_values,
        "pit": pit,                                           # F(y_true); ~Uniform(0,1) if calibrated
    })
    for q in sorted(quantile_cols):                           # predictive quantiles, hours
        per_event[f"q{int(round(q * 100)):02d}"] = quantile_cols[q]
    per_event["sla_pred_hit"] = predicted_hit
    per_event["sla_actual_hit"] = actual_hit
    per_event["baseline_pred"] = (np.asarray(baseline_predictions, dtype=float)
                                  if baseline_predictions is not None else np.nan)
    if timestamps is not None and event_positions is not None:
        pos = np.asarray(event_positions)
        if len(pos) != n:
            raise ValueError(f"{source_id}/{model_name}: {len(pos)} event positions for {n} predictions")
        ts = pd.Series(timestamps).reset_index(drop=True)
        per_event.insert(0, "event_index", pos)
        per_event.insert(1, "prev_arrival_timestamp", ts.iloc[pos - 1].to_numpy())
        per_event.insert(2, "arrival_timestamp", ts.iloc[pos].to_numpy())

    return SourceMetrics(
        source_id=source_id,
        model=model_name,
        n_test_events=n,
        low_n=n < low_n_threshold,
        mean_nll=float(mean_nll),
        mean_crps=mean_crps,
        mae=mae_val,
        rmse=rmse_val,
        coverage=coverage,
        sla=sla_result,
        sla_threshold_hours=sla_hours,
        baseline_mae=baseline_mae_val,
        mae_pct_improvement=pct_improve,
        per_event=per_event,
    )


def source_metrics_to_row(m: SourceMetrics) -> dict:
    """Flatten a SourceMetrics into one dict, suitable for
    pd.DataFrame(list_of_rows) — used to build the notebook's comparison
    table."""
    row = {
        "source_id": m.source_id,
        "model": m.model,
        "n_test_events": m.n_test_events,
        "low_n": m.low_n,
        "mean_nll": m.mean_nll,
        "mean_crps": m.mean_crps,
        "mae": m.mae,
        "rmse": m.rmse,
        "sla_accuracy": m.sla.accuracy,
        "sla_precision": m.sla.precision,
        "sla_recall": m.sla.recall,
        "sla_f1": m.sla.f1,
        "sla_threshold_hours": m.sla_threshold_hours,
        "baseline_mae": m.baseline_mae,
        "mae_pct_improvement": m.mae_pct_improvement,
    }
    for level, cov in m.coverage.items():
        row[f"coverage_nominal_{int(level*100)}"] = level
        row[f"coverage_observed_{int(level*100)}"] = cov.observed_coverage
    return row
