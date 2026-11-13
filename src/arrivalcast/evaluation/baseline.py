"""
Naive baseline: "predict the same gap as the immediately preceding one."

Deliberately dumber than HomogeneousPoisson (which is already the MLE
version of "predict the historical mean gap") — Poisson is one of the four
real models under evaluation, so comparing against it as the "naive"
floor would be circular. This lag-1 heuristic needs no fitting at all,
giving every model (Poisson included) a common non-statistical floor to
beat.
"""
from __future__ import annotations

import numpy as np


def lag1_predictions(gaps_hours: np.ndarray) -> np.ndarray:
    """
    predicted[i] = gaps_hours[i-1], i.e. "next gap will equal the previous
    gap." gaps_hours should span the full chronological history up through
    whatever split you're scoring (e.g. concatenated train+val+test gaps),
    so that the first TEST prediction is still grounded in a real
    preceding gap (the last gap of train/val) rather than undefined.

    Returns an array the same length as gaps_hours; index 0 has no valid
    predecessor and is returned as NaN — callers scoring only a test-index
    slice should simply never index into position 0 of the full sequence.
    """
    pred = np.empty(len(gaps_hours))
    pred[0] = np.nan
    pred[1:] = gaps_hours[:-1]
    return pred
