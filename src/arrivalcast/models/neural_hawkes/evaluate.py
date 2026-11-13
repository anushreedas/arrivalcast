"""
Evaluate a trained NeuralHawkes model on one source's test split.

Same "replay the full chronological history once, then read off state at
each test position" principle as models/lognormmix/evaluate.py and
models/hawkes/evaluate.py — here the state is the CTLSTM's (c, c_bar, o,
decay_rate) after each real event, obtained via NeuralHawkes.forward()'s
`all_states` return (see model.py) rather than a closed-form recursion
(Hawkes) or a plain RNN hidden state (LogNormMix).

Requires a model already trained on the train split (fitting happens in
the notebook) — this file only evaluates.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from arrivalcast.core.splits import chronological_split
from arrivalcast.core.windows import compute_window
from arrivalcast.evaluation.baseline import lag1_predictions
from arrivalcast.evaluation.evaluate import PerEventPrediction, SourceMetrics, aggregate_source_metrics
from arrivalcast.features.calendar_features import compute_calendar_features
from arrivalcast.features.time2vec import Time2Vec
from arrivalcast.models.neural_hawkes.dataloader import build_event_inputs
from arrivalcast.models.neural_hawkes.model import NeuralHawkes
from arrivalcast.models.neural_hawkes.predictive import NeuralHawkesPredictive


@torch.no_grad()
def evaluate_nhp_source(
    source_id: str,
    timestamps: pd.Series,
    global_study_end: pd.Timestamp,
    model: NeuralHawkes,
    time2vec: Time2Vec,
    use_calendar_features: bool = True,
    sla_hours: float = 24.0,
    low_n_threshold: int = 5,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    n_quad_points: int = 300,
) -> SourceMetrics | None:
    """
    model, time2vec : already trained on this source's train split (see
        the notebook's train_nhp_one_source) — evaluation-only here,
        mirroring models/lognormmix/evaluate.py's evaluate_lognormmix_source.
    """
    model.eval()
    ts = pd.Series(timestamps).sort_values().reset_index(drop=True)
    idx = chronological_split(len(ts), train_frac, val_frac)
    if len(idx.train) < 5 or len(idx.test) == 0:
        return None
    test_start = idx.test[0]
    if test_start < 1:
        return None

    full_window = compute_window(ts, study_start=ts.iloc[0], study_end=global_study_end)
    # NHP's own convention (see models/neural_hawkes/dataloader.py) prepends
    # the trivial first_gap_hours (always 0, since t0 == first event by
    # construction) so gaps_full has one entry PER EVENT, aligned 1:1 with
    # events (unlike Hawkes/Poisson/LogNormMix, which use N-1 inter-event
    # gaps) — replicate that here for consistency with build_event_inputs.
    gaps_full = np.concatenate(([full_window.first_gap_hours], full_window.gaps_hours))

    if use_calendar_features:
        cal_full = compute_calendar_features(ts).to_numpy()
    else:
        cal_full = np.zeros((len(gaps_full), 0))

    event_inputs = build_event_inputs(
        gaps_full, cal_full, time2vec, float(model.gap_scale)
    )  # (N, input_dim)
    inputs_t = event_inputs.unsqueeze(0)  # (1, N, input_dim)
    gaps_t = torch.tensor(gaps_full, dtype=torch.float32).unsqueeze(0)  # (1, N)
    # Integration samples aren't used for anything here (we only need
    # per-step state, not the training log-likelihood), so pass a
    # minimal dummy tensor of the right shape.
    dummy_integration = torch.zeros(1, len(gaps_full), 1)

    _, _, _, all_states = model.forward(inputs_t, gaps_t, dummy_integration)
    c_all, c_bar_all, o_all, decay_all = (s.squeeze(0) for s in all_states)  # (N, hidden_dim)

    # State after event i (0-indexed) is what predicts the GAP TO event
    # i+1 — i.e. to predict gaps_full[k] (the gap ending at event k, for
    # k = test_start .. N-1), condition on the state after event k-1.
    test_event_positions = np.arange(test_start, len(gaps_full))
    test_gaps = gaps_full[test_event_positions]
    state_positions = test_event_positions - 1

    baseline_full = lag1_predictions(gaps_full)
    baseline_test = baseline_full[test_event_positions]

    predictions = []
    for g, pos in zip(test_gaps, state_positions):
        state = (c_all[pos], c_bar_all[pos], o_all[pos], decay_all[pos])
        pred = NeuralHawkesPredictive(model, state, n_quad_points=n_quad_points)
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
        model_name="NeuralHawkes",
        predictions=predictions,
        sla_hours=sla_hours,
        low_n_threshold=low_n_threshold,
        baseline_predictions=baseline_test,
        timestamps=ts,
        event_positions=np.arange(test_start, len(ts)),
    )
