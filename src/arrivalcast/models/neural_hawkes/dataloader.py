"""
Dataloader for the Neural Hawkes Process.

Unlike Poisson/Hawkes, this model needs a genuinely different input shape:
a per-event input vector (not just a scalar gap), a set of Monte Carlo
integration sample times per interval, and padding/masking to batch
sources of wildly different lengths (17,544 events for SRC_08 vs. 2 for
SRC_23) together.

Feature composition per event is: Time2Vec(Δt) optionally concatenated
with calendar features of the event's own timestamp. Both are legitimate
inputs here (unlike for Poisson/Hawkes) because the CTLSTM's input slot is
free-form — see features/calendar_features.py and features/time2vec.py.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from arrivalcast.core.splits import get_chronological_windows
from arrivalcast.features.calendar_features import compute_calendar_features

if TYPE_CHECKING:  # torch is imported lazily at runtime
    from arrivalcast.features.time2vec import Time2Vec

# torch (and the Time2Vec module, which depends on it) is only needed by the
# batching helpers below (build_event_inputs / pad_batch), not by
# NeuralHawkesDataLoader itself — imported lazily inside those functions so
# the pure numpy/pandas splitting logic above can be used/tested in
# environments without torch installed.


@dataclass
class NHPSequence:
    """One source's prepared sequence (unbatched, unpadded)."""

    gaps_hours: np.ndarray  # (n_events,) Δt_i, first entry = t_1 - t0
    calendar_features: np.ndarray  # (n_events, n_calendar_feats)
    tail_hours: float
    n_events: int


class NeuralHawkesDataLoader:
    def __init__(
        self,
        timestamps: pd.Series,
        train_frac: float = 0.7,
        val_frac: float = 0.15,
        use_calendar_features: bool = True,
        n_mc_samples: int = 5,
        seed: int = 0,
        global_study_end: pd.Timestamp | None = None,
    ):
        """
        global_study_end : the real end of the observation period for this
            dataset (e.g. the fixed generation cutoff for the synthetic
            data, or the BTS extraction cutoff date) — NOT inferred. This
            is the boundary for the LAST split only (normally test);
            train/val always use the next split's first event as their
            bound regardless of this value, since their "responsibility"
            ends where the next split's data begins, not at the dataset's
            true end. If omitted, the last split falls back to its own
            last event, which understates its right-censored tail — pass
            this whenever you know the real cutoff (you almost always do).
        """
        self.timestamps = pd.Series(timestamps).sort_values().reset_index(drop=True)
        self.train_frac = train_frac
        self.val_frac = val_frac
        self.use_calendar_features = use_calendar_features
        self.n_mc_samples = n_mc_samples
        self.rng = np.random.default_rng(seed)
        self.global_study_end = global_study_end

    def _sequence_from_window(self, ts: pd.Series, window) -> NHPSequence:
        gaps = np.concatenate(([window.first_gap_hours], window.gaps_hours))
        if self.use_calendar_features:
            cal = compute_calendar_features(ts).to_numpy()
        else:
            cal = np.zeros((len(ts), 0))
        return NHPSequence(
            gaps_hours=gaps,
            calendar_features=cal,
            tail_hours=window.tail_hours,
            n_events=window.n_events,
        )

    def get_splits(self) -> dict[str, NHPSequence]:
        """
        Split chronologically and compute each split's window via
        core.splits.get_chronological_windows (shared boundary logic
        across every model — see that function's docstring for the full
        explanation of study_end, and always pass global_study_end).
        """
        splits_raw = get_chronological_windows(
            self.timestamps, self.train_frac, self.val_frac, self.global_study_end
        )
        splits = {}
        for name, sw in splits_raw.items():
            if sw is None:
                splits[name] = NHPSequence(np.array([]), np.zeros((0, 0)), 0.0, 0)
                continue
            splits[name] = self._sequence_from_window(sw.timestamps, sw.window)
        return splits

    def sample_integration_times(self, gaps_hours: np.ndarray) -> np.ndarray:
        """
        Draw n_mc_samples uniform points within each interval, measured as
        time-since-the-start-of-that-interval (i.e. in [0, gap_i)) — this
        is what NeuralHawkes.forward() expects for `integration_gaps`.
        """
        n = len(gaps_hours)
        u = self.rng.uniform(0.0, 1.0, size=(n, self.n_mc_samples))
        return u * gaps_hours[:, None]

    def sample_tail_integration_times(self, tail_hours: float) -> np.ndarray:
        u = self.rng.uniform(0.0, 1.0, size=(self.n_mc_samples,))
        return u * tail_hours


def build_event_inputs(
    gaps_hours: np.ndarray,
    calendar_features: np.ndarray,
    time2vec: Time2Vec,
    gap_scale: float = 1.0,
):
    """
    Construct the per-event input vector fed to NeuralHawkes: Time2Vec(Δt)
    concatenated with the event's calendar features.

    gaps_hours : (n_events,) numpy array
    calendar_features : (n_events, n_calendar_feats) numpy array (may have
        0 columns if calendar features are disabled)
    time2vec : a Time2Vec module (shared across sources/models so its
        learned frequencies transfer, or a fresh one per model — your call)

    Returns
    -------
    torch.Tensor, shape (n_events, time2vec.out_dim + n_calendar_feats)
    """
    import torch

    # FIX: Time2Vec gets the gap in units of the source's typical gap
    # (gap_scale = train median gap, see NeuralHawkes.gap_scale). Raw hours
    # (up to ~2,000 for quarterly sources) pushed the linear Time2Vec term to
    # huge values and the sine terms into aliasing. The CTLSTM decay and the
    # intensity still work in hours; only this input feature is rescaled.
    gaps_t = torch.tensor(np.asarray(gaps_hours) / float(gap_scale), dtype=torch.float32)
    t2v = time2vec(gaps_t)  # (n_events, out_dim)
    if calendar_features.shape[1] > 0:
        cal_t = torch.tensor(calendar_features, dtype=torch.float32)
        return torch.cat([t2v, cal_t], dim=-1)
    return t2v


def pad_batch(
    sequences: list[NHPSequence],
    time2vec: Time2Vec,
    n_mc_samples: int,
    rng: np.random.Generator,
    gap_scale: float = 1.0,
):
    """
    Collate a list of NHPSequence (one per source) into padded batch
    tensors ready for NeuralHawkes.forward()/log_likelihood().

    Returns a dict with keys: inputs, gaps, integration_gaps, mask,
    lengths, tail_hours, tail_integration_samples — see NeuralHawkes
    docstrings for shapes/semantics of each.
    """
    import torch

    lengths = [seq.n_events for seq in sequences]
    max_len = max(lengths) if lengths else 0
    batch = len(sequences)
    input_dim = time2vec.out_dim + (sequences[0].calendar_features.shape[1] if sequences else 0)

    inputs = torch.zeros(batch, max_len, input_dim)
    gaps = torch.zeros(batch, max_len)
    integration_gaps = torch.zeros(batch, max_len, n_mc_samples)
    mask = torch.zeros(batch, max_len)
    tail_hours = torch.zeros(batch)
    tail_integration_samples = torch.zeros(batch, n_mc_samples)

    for b, seq in enumerate(sequences):
        n = seq.n_events
        if n == 0:
            continue
        ev_inputs = build_event_inputs(seq.gaps_hours, seq.calendar_features, time2vec, gap_scale)
        inputs[b, :n] = ev_inputs
        gaps[b, :n] = torch.tensor(seq.gaps_hours, dtype=torch.float32)
        u = rng.uniform(0.0, 1.0, size=(n, n_mc_samples))
        integration_gaps[b, :n] = torch.tensor(u * seq.gaps_hours[:, None], dtype=torch.float32)
        mask[b, :n] = 1.0
        tail_hours[b] = seq.tail_hours
        u_tail = rng.uniform(0.0, 1.0, size=(n_mc_samples,))
        tail_integration_samples[b] = torch.tensor(u_tail * seq.tail_hours, dtype=torch.float32)

    return {
        "inputs": inputs,
        "gaps": gaps,
        "integration_gaps": integration_gaps,
        "mask": mask,
        "lengths": torch.tensor(lengths, dtype=torch.long),
        "tail_hours": tail_hours,
        "tail_integration_samples": tail_integration_samples,
    }
