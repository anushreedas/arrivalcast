"""
Chronological train/val/test splitting for a single source's event sequence.

Point-process data cannot be split randomly — a model that "sees" an event
out of order effectively cheats by conditioning on the future. Instead we
split by position along the sequence: the first train_frac of events (by
count, not by wall-clock time) form train, the next val_frac form
validation, and the remainder is test. Splitting by event count rather than
by calendar time keeps sparse sources (monthly/quarterly/annual) from
ending up with an empty validation or test set.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class SplitIndices:
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray


@dataclass
class SplitWindow:
    """One split's ObservationWindow plus its own raw timestamps, so
    downstream dataloaders (e.g. models/neural_hawkes, which also needs
    calendar features per timestamp) never have to re-derive which rows
    belong to this split from window boundaries alone."""

    window: object  # core.windows.ObservationWindow (avoiding an
    # eager import here keeps this module free of any dependency direction
    # issues; get_chronological_windows imports it locally when needed)
    timestamps: pd.Series


def chronological_split(
    n_events: int,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
) -> SplitIndices:
    """
    Return index arrays (into the ordered event sequence) for train/val/test.

    Boundaries are computed on event *count*. Train always gets at least one
    event, and validation gets one whenever at least two events remain after
    train, so very sparse sources (e.g. a quarterly source with 8 events) still
    produce non-empty train/val/test splits. (For n_events <= 5 validation or
    test can be empty; such sources are skipped by every model's evaluator.)
    """
    if not 0 < train_frac < 1 or not 0 <= val_frac < 1 or train_frac + val_frac >= 1:
        raise ValueError("require 0 < train_frac < 1, 0 <= val_frac, train+val < 1")

    idx = np.arange(n_events)
    n_train = max(int(round(n_events * train_frac)), 1 if n_events > 0 else 0)
    n_val = max(int(round(n_events * val_frac)), 1 if n_events - n_train > 1 else 0)
    n_train = min(n_train, n_events)
    n_val = min(n_val, n_events - n_train)

    return SplitIndices(
        train=idx[:n_train],
        val=idx[n_train : n_train + n_val],
        test=idx[n_train + n_val :],
    )


def split_timestamps(
    timestamps: pd.Series,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Convenience wrapper: split a sorted timestamp Series directly."""
    ts = pd.Series(timestamps).sort_values().reset_index(drop=True)
    idx = chronological_split(len(ts), train_frac, val_frac)
    return ts.iloc[idx.train], ts.iloc[idx.val], ts.iloc[idx.test]


def get_chronological_windows(
    timestamps: pd.Series,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    global_study_end: pd.Timestamp | None = None,
):
    """
    Single source of truth for "what study_end should each split use?",
    used by every model's dataloader (models/poisson, models/hawkes,
    models/neural_hawkes, models/lognormmix) instead of each
    re-implementing this boundary logic — they diverged subtly when
    implemented separately, which this consolidation fixes.

    Convention: a split's study_end is the timestamp of the first event of
    the NEXT split (so its right-censored tail is exactly "time until the
    next split's data begins," not an arbitrary cutoff). The final
    non-empty split (normally test) has no next split to bound it, so it
    uses `global_study_end` if provided — the real end of the observation
    period (dataset generation cutoff / extraction date). Without it, the
    final split falls back to its own last event, which collapses its
    tail to zero; always pass global_study_end when you know the real
    cutoff (see core/windows.py docstring on why this matters).

    Returns
    -------
    dict[str, SplitWindow | None] — keys "train", "val", "test". A split
    with zero events maps to None rather than a degenerate window.
    SplitWindow.timestamps holds exactly that split's own event
    timestamps (not the next split's boundary event), so downstream
    dataloaders never need to re-derive which rows belong to which split.
    """
    from arrivalcast.core.windows import compute_window  # local import: avoids a
    # circular import, since windows.py has no dependency on splits.py
    # but this function lives in splits.py for API discoverability.

    ts = pd.Series(timestamps).sort_values().reset_index(drop=True)
    idx = chronological_split(len(ts), train_frac, val_frac)
    order = [("train", idx.train), ("val", idx.val), ("test", idx.test)]
    non_empty = [(name, split_idx) for name, split_idx in order if len(split_idx) > 0]

    bounds = {}
    for i, (name, split_idx) in enumerate(non_empty):
        if i + 1 < len(non_empty):
            bounds[name] = ts.iloc[non_empty[i + 1][1][0]]
        elif global_study_end is not None:
            bounds[name] = global_study_end
        else:
            bounds[name] = ts.iloc[split_idx[-1]]

    windows = {}
    for name, split_idx in order:
        if len(split_idx) == 0:
            windows[name] = None
            continue
        split_ts = ts.iloc[split_idx].reset_index(drop=True)
        window = compute_window(split_ts, study_start=split_ts.iloc[0], study_end=bounds[name])
        windows[name] = SplitWindow(window=window, timestamps=split_ts)
    return windows
