"""
Fit HawkesExponential on train and evaluate every metric on test, for one
source.

Unlike Poisson, Hawkes predictions depend on history via the excitation
state R — R must be replayed across ALL prior events (train, then val,
then test-so-far) before scoring each test event, not reset at the start
of test. See models/hawkes/predictive.py for the closed-form survival
function this state feeds into.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from arrivalcast.core.splits import chronological_split
from arrivalcast.core.windows import compute_window
from arrivalcast.evaluation.baseline import lag1_predictions
from arrivalcast.evaluation.evaluate import PerEventPrediction, SourceMetrics, aggregate_source_metrics
from arrivalcast.models.hawkes.model import HawkesExponential
from arrivalcast.models.hawkes.predictive import HawkesPredictive, compute_R_trajectory


def evaluate_hawkes_source(
    source_id: str,
    timestamps: pd.Series,
    global_study_end: pd.Timestamp,
    sla_hours: float = 24.0,
    low_n_threshold: int = 5,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
) -> SourceMetrics | None:
    ts = pd.Series(timestamps).sort_values().reset_index(drop=True)
    idx = chronological_split(len(ts), train_frac, val_frac)
    if len(idx.train) < 5 or len(idx.test) == 0:
        return None
    test_start = idx.test[0]
    if test_start < 1:
        return None

    train_end_bound = ts.iloc[idx.val[0]] if len(idx.val) else (
        ts.iloc[idx.test[0]] if len(idx.test) else global_study_end
    )
    train_ts = ts.iloc[idx.train]
    train_window = compute_window(train_ts, study_start=train_ts.iloc[0], study_end=train_end_bound)
    model = HawkesExponential().fit(train_window.gaps_hours, train_window.total_hours)
    params = model.params

    # Full gap sequence across the whole chronological history, so R can
    # be replayed continuously (see module docstring) — same convention
    # as models/poisson/evaluate.py.
    full_window = compute_window(ts, study_start=ts.iloc[0], study_end=global_study_end)
    gaps_full = full_window.gaps_hours

    # R_before[i] is the excitation state used to predict gaps_full[i]
    # (i.e. conditioned on every event strictly before it, starting from
    # R0=0 at the very first event of the source's whole history).
    R_before, _ = compute_R_trajectory(gaps_full, params.beta, R0=0.0)

    test_gaps = gaps_full[test_start - 1 :]
    test_R = R_before[test_start - 1 :]

    baseline_full = lag1_predictions(gaps_full)
    baseline_test = baseline_full[test_start - 1 :]

    predictions = []
    for g, R in zip(test_gaps, test_R):
        pred = HawkesPredictive(params, R)
        predictions.append(
            PerEventPrediction(
                y_true=float(g),
                cdf_fn=pred.cdf,
                log_prob=pred.log_prob(float(g)),
                t_max=pred.suggested_t_max(),
            )
        )

    return aggregate_source_metrics(
        source_id=source_id,
        model_name="Hawkes",
        predictions=predictions,
        sla_hours=sla_hours,
        low_n_threshold=low_n_threshold,
        baseline_predictions=baseline_test,
        timestamps=ts,
        event_positions=np.arange(test_start, len(ts)),
    )
