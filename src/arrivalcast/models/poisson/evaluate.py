"""
Fit HomogeneousPoisson on train and evaluate every metric on test, for one
source. Poisson is memoryless/stateless, so this is the simplest of the
four evaluate.py files — no history replay needed, unlike Hawkes/NHP/LNM.

Scoring works off ONE continuous gap sequence spanning the full
chronological history (train+val+test), not per-split local gaps — this
matters because the first test event's arrival gap is measured from the
last pre-test event (in val, or train if val is empty), and a source whose
test split has only 1 event still has exactly that one real gap to score
(its own internal, split-local gap array would be empty, since you need >=2
events in a split to have an internal diff).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from arrivalcast.core.splits import chronological_split
from arrivalcast.core.windows import compute_window
from arrivalcast.evaluation.baseline import lag1_predictions
from arrivalcast.evaluation.evaluate import PerEventPrediction, SourceMetrics, aggregate_source_metrics
from arrivalcast.models.poisson.model import HomogeneousPoisson
from arrivalcast.models.poisson.predictive import PoissonPredictive


def evaluate_poisson_source(
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
    if len(idx.train) < 2 or len(idx.test) == 0:
        return None
    test_start = idx.test[0]  # position of the first test event in the full series
    if test_start < 1:
        return None  # need at least one preceding event to define its arrival gap

    # Fit on train only, using train's own local window (bounded by val's
    # first event, or global_study_end if val is empty).
    train_end_bound = ts.iloc[idx.val[0]] if len(idx.val) else (
        ts.iloc[idx.test[0]] if len(idx.test) else global_study_end
    )
    train_ts = ts.iloc[idx.train]
    train_window = compute_window(train_ts, study_start=train_ts.iloc[0], study_end=train_end_bound)
    model = HomogeneousPoisson().fit(train_window.n_events, train_window.total_hours)
    pred = PoissonPredictive(model)

    # One continuous gap sequence over the FULL chronological history, so
    # test_start's own gap (measured from the last pre-test event) is
    # included — see module docstring.
    full_window = compute_window(ts, study_start=ts.iloc[0], study_end=global_study_end)
    gaps_full = full_window.gaps_hours  # gaps_full[i-1] = arrival gap of event i (1-indexed)

    test_gaps = gaps_full[test_start - 1 :]  # one gap per test event, in order
    baseline_full = lag1_predictions(gaps_full)
    baseline_test = baseline_full[test_start - 1 :]

    predictions = [
        PerEventPrediction(
            y_true=float(g),
            cdf_fn=pred.cdf,
            log_prob=pred.log_prob(float(g)),
            t_max=pred.suggested_t_max(),
        )
        for g in test_gaps
    ]

    return aggregate_source_metrics(
        source_id=source_id,
        model_name="Poisson",
        predictions=predictions,
        sla_hours=sla_hours,
        low_n_threshold=low_n_threshold,
        baseline_predictions=baseline_test,
        timestamps=ts,
        event_positions=np.arange(test_start, len(ts)),
    )
