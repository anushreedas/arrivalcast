"""
Evaluate a trained LogNormMix model on one source's test split.

Unlike Hawkes (closed-form state via compute_R_trajectory) or Poisson
(stateless), LogNormMix's conditioning state is a GRU hidden state that
only exists by literally running the RNN forward. So this file replays the
FULL chronological input sequence (train+val+test) through the trained
model once, and reads off the hidden state at each position needed to
score a test-split prediction — the same "warm up the recurrent state
through everything before the test split" principle as Hawkes's R replay,
just implemented via a forward pass instead of a closed-form recursion.

Requires a model already trained on the train split (see
models/lognormmix/model.py / the notebook's training loop) — this file
only evaluates, it does not fit.
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
from arrivalcast.models.lognormmix.dataloader import LogNormMixSequence, build_step_inputs
from arrivalcast.models.lognormmix.model import LogNormMix
from arrivalcast.models.lognormmix.predictive import LogNormMixPredictive


@torch.no_grad()
def evaluate_lognormmix_source(
    source_id: str,
    timestamps: pd.Series,
    global_study_end: pd.Timestamp,
    model: LogNormMix,
    use_calendar_features: bool = True,
    sla_hours: float = 24.0,
    low_n_threshold: int = 5,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
) -> SourceMetrics | None:
    """
    model : a LogNormMix already trained on this source's train split
        (fitting happens in the notebook, not here — this function is
        evaluation-only, mirroring how models/hawkes/evaluate.py takes an
        already-fitted HawkesExponential's params rather than re-fitting).
    """
    model.eval()
    ts = pd.Series(timestamps).sort_values().reset_index(drop=True)
    idx = chronological_split(len(ts), train_frac, val_frac)
    if len(idx.train) < 5 or len(idx.test) == 0:
        return None
    test_start = idx.test[0]
    if test_start < 2:
        # LogNormMix can never predict the very first gap of a source (no
        # history to condition the RNN on), so test_start must be >= 2 —
        # see models/lognormmix/dataloader.py module docstring.
        return None

    full_window = compute_window(ts, study_start=ts.iloc[0], study_end=global_study_end)
    gaps_full = full_window.gaps_hours  # length N-1

    if use_calendar_features:
        cal_full = compute_calendar_features(ts).to_numpy()[1:]  # align to gaps
    else:
        cal_full = np.zeros((len(gaps_full), 0))

    full_seq = LogNormMixSequence(
        gaps_hours=gaps_full, calendar_features=cal_full,
        tail_hours=full_window.tail_hours, n_gaps=len(gaps_full),
    )
    inputs_full, targets_full = build_step_inputs(full_seq)
    # inputs_full[j] -> predicts targets_full[j] = gaps_full[j+1]

    inputs_t = torch.tensor(inputs_full, dtype=torch.float32).unsqueeze(0)  # (1, L, D)
    hidden_states = model.forward(inputs_t).squeeze(0)  # (L, hidden_dim)

    # Test targets are gaps_full[test_start-1 : N-1], i.e. targets_full
    # indices [test_start-2 : len(targets_full)] (see module docstring
    # derivation in the corresponding Hawkes/Poisson evaluators for the
    # same off-by-one reasoning).
    j_start = test_start - 2
    test_targets = targets_full[j_start:]
    test_hidden = hidden_states[j_start:]

    baseline_full = lag1_predictions(gaps_full)
    baseline_test = baseline_full[test_start - 1 :]

    predictions = []
    for y, h in zip(test_targets, test_hidden):
        log_weights, means, log_stds = model.decoder.get_params(h.unsqueeze(0))
        pred = LogNormMixPredictive(
            weights=log_weights.exp().squeeze(0).numpy(),
            means=means.squeeze(0).numpy(),
            stds=log_stds.exp().squeeze(0).numpy(),
        )
        predictions.append(
            PerEventPrediction(
                y_true=float(y),
                cdf_fn=pred.cdf,
                log_prob=pred.log_prob(float(y)),
                t_max=pred.suggested_t_max(),
            )
        )

    return aggregate_source_metrics(
        source_id=source_id,
        model_name="LogNormMix",
        predictions=predictions,
        sla_hours=sla_hours,
        low_n_threshold=low_n_threshold,
        baseline_predictions=baseline_test,
        timestamps=ts,
        event_positions=np.arange(test_start, len(ts)),
    )
