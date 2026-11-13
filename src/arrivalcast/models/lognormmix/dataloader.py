"""
Dataloader for LogNormMix.

Input convention: at step i, the RNN consumes log(Δt_i) (plus optional
covariates of event i) and its resulting hidden state is used to predict
Δt_{i+1}. Concretely, for a sequence of n gaps Δt_1..Δt_n:

    inputs[i]       = features of gap Δt_i,      i = 1..n-1   (n-1 steps)
    target_gaps[i]  = Δt_{i+1},                  i = 1..n-1
    tail target     = the right-censored interval after Δt_n (survival term)

So an n-gap sequence yields (n-1) supervised predictions plus one survival
term — Δt_1 itself is "spent" providing the first input and is never a
prediction target (there's no history before it to condition on).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from arrivalcast.core.splits import get_chronological_windows
from arrivalcast.features.calendar_features import compute_calendar_features


@dataclass
class LogNormMixSequence:
    gaps_hours: np.ndarray  # (n_gaps,) Δt_1..Δt_n, chronological
    calendar_features: np.ndarray  # (n_gaps, n_calendar_feats), aligned to the
    # event that ENDS each gap (i.e. features[i] describes the arrival that
    # produced gaps_hours[i])
    tail_hours: float
    n_gaps: int


class LogNormMixDataLoader:
    def __init__(
        self,
        timestamps: pd.Series,
        train_frac: float = 0.7,
        val_frac: float = 0.15,
        use_calendar_features: bool = True,
        global_study_end: pd.Timestamp | None = None,
    ):
        """
        global_study_end : the real end of the observation period for this
            dataset — see NeuralHawkesDataLoader.__init__ for the full
            explanation. Used only for the last chronological split
            (normally test); train/val always bound themselves by the next
            split's first event regardless of this value.
        """
        self.timestamps = pd.Series(timestamps).sort_values().reset_index(drop=True)
        self.train_frac = train_frac
        self.val_frac = val_frac
        self.use_calendar_features = use_calendar_features
        self.global_study_end = global_study_end

    def _sequence_from_window(self, ts: pd.Series, window) -> LogNormMixSequence:
        # gaps_hours excludes the trivial first_gap (t0 == first event by
        # construction, so first_gap_hours is always 0) — LogNormMix only
        # ever conditions on real inter-arrival gaps.
        gaps = window.gaps_hours
        if self.use_calendar_features:
            cal = compute_calendar_features(ts).to_numpy()[1:]  # align to gaps, not events
        else:
            cal = np.zeros((len(gaps), 0))
        return LogNormMixSequence(
            gaps_hours=gaps,
            calendar_features=cal,
            tail_hours=window.tail_hours,
            n_gaps=len(gaps),
        )

    def get_splits(self) -> dict[str, LogNormMixSequence]:
        """Window boundaries come from the shared
        core.splits.get_chronological_windows helper — see its
        docstring for how the final split's tail_hours is bounded by
        global_study_end (if you passed one) instead of falling back to
        its own last event."""
        splits_raw = get_chronological_windows(
            self.timestamps, self.train_frac, self.val_frac, self.global_study_end
        )
        splits = {}
        for name, sw in splits_raw.items():
            if sw is None or sw.window.n_events < 2:
                splits[name] = LogNormMixSequence(np.array([]), np.zeros((0, 0)), 0.0, 0)
                continue
            splits[name] = self._sequence_from_window(sw.timestamps, sw.window)
        return splits


def build_step_inputs(seq: LogNormMixSequence):
    """
    Build (inputs, target_gaps) for one sequence, following the offset
    convention documented at module level: inputs are gaps 1..n-1, targets
    are gaps 2..n. Returns plain numpy arrays; batching/padding into torch
    tensors happens in pad_batch() (torch imported lazily there so this
    function has no torch dependency).
    """
    if seq.n_gaps < 2:
        n_feat = 1 + seq.calendar_features.shape[1]
        return np.zeros((0, n_feat)), np.zeros((0,))

    log_gaps = np.log(np.clip(seq.gaps_hours, 1e-8, None)).reshape(-1, 1)
    if seq.calendar_features.shape[1] > 0:
        features = np.concatenate([log_gaps, seq.calendar_features], axis=1)
    else:
        features = log_gaps

    inputs = features[:-1]  # steps 1..n-1
    target_gaps = seq.gaps_hours[1:]  # gaps 2..n
    return inputs, target_gaps


def build_terminal_input(seq: LogNormMixSequence) -> np.ndarray:
    """
    Feature vector of the LAST observed gap Δt_n — the one extra RNN input
    needed so the tail-survival term (and the next-gap forecast) is
    conditioned on a state that has actually consumed Δt_n. build_step_inputs
    stops at Δt_{n-1}, so the state at its last step is one gap behind.
    Returns zeros if the sequence has no gaps.
    """
    n_feat = 1 + seq.calendar_features.shape[1]
    if seq.n_gaps < 1:
        return np.zeros(n_feat)
    log_gap = np.log(np.clip(seq.gaps_hours[-1:], 1e-8, None))
    if seq.calendar_features.shape[1] > 0:
        return np.concatenate([log_gap, seq.calendar_features[-1]])
    return log_gap


def pad_batch(sequences: list[LogNormMixSequence]):
    """
    Collate a list of LogNormMixSequence into padded batch tensors ready
    for LogNormMix.log_likelihood().

    Returns a dict with keys: inputs, target_gaps, mask, lengths,
    tail_hours, terminal_inputs — see LogNormMix.log_likelihood() for shapes/semantics.
    """
    import torch

    prepared = [build_step_inputs(seq) for seq in sequences]
    lengths = [inp.shape[0] for inp, _ in prepared]
    max_len = max(lengths) if lengths else 0
    batch = len(sequences)
    input_dim = prepared[0][0].shape[1] if prepared and prepared[0][0].shape[0] > 0 else (
        1 + sequences[0].calendar_features.shape[1] if sequences else 1
    )

    inputs = torch.zeros(batch, max_len, input_dim)
    target_gaps = torch.zeros(batch, max_len)
    mask = torch.zeros(batch, max_len)
    tail_hours = torch.zeros(batch)
    terminal_inputs = torch.zeros(batch, input_dim)

    for b, ((inp, tgt), seq) in enumerate(zip(prepared, sequences)):
        n = inp.shape[0]
        terminal_inputs[b] = torch.tensor(build_terminal_input(seq), dtype=torch.float32)
        if n == 0:
            tail_hours[b] = seq.tail_hours
            continue
        inputs[b, :n] = torch.tensor(inp, dtype=torch.float32)
        target_gaps[b, :n] = torch.tensor(tgt, dtype=torch.float32)
        mask[b, :n] = 1.0
        tail_hours[b] = seq.tail_hours

    return {
        "inputs": inputs,
        "target_gaps": target_gaps,
        "mask": mask,
        "lengths": torch.tensor(lengths, dtype=torch.long),
        "tail_hours": tail_hours,
        "terminal_inputs": terminal_inputs,
    }
