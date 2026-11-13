"""
Dataloader for the Hawkes baseline.

Like Poisson, Hawkes needs no covariates — just the gap sequence and the
window length per split, used directly by HawkesExponential.fit() /
.log_likelihood(). Window boundaries come from the shared
core.splits.get_chronological_windows helper (also used by
models/poisson, models/neural_hawkes, models/lognormmix) rather than
being recomputed here.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from arrivalcast.core.splits import get_chronological_windows


@dataclass
class HawkesSplit:
    gaps_hours: np.ndarray
    total_hours: float
    n_events: int


class HawkesDataLoader:
    def __init__(
        self,
        timestamps: pd.Series,
        train_frac: float = 0.7,
        val_frac: float = 0.15,
        global_study_end: pd.Timestamp | None = None,
    ):
        """
        global_study_end : real end of the observation period, used to
            bound the LAST split only — see
            core.splits.get_chronological_windows for the full
            explanation (identical convention across every model).
        """
        self.timestamps = pd.Series(timestamps).sort_values().reset_index(drop=True)
        self.train_frac = train_frac
        self.val_frac = val_frac
        self.global_study_end = global_study_end

    def get_splits(self) -> dict[str, HawkesSplit]:
        splits_raw = get_chronological_windows(
            self.timestamps, self.train_frac, self.val_frac, self.global_study_end
        )
        splits = {}
        for name, sw in splits_raw.items():
            if sw is None or sw.window.n_events < 2:
                n = sw.window.n_events if sw is not None else 0
                splits[name] = HawkesSplit(np.array([]), 0.0, n)
                continue
            window = sw.window
            splits[name] = HawkesSplit(
                gaps_hours=window.gaps_hours,
                total_hours=max(window.total_hours, 1e-6),
                n_events=window.n_events,
            )
        return splits
