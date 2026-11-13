"""
Dataloader for the Poisson baseline.

The homogeneous Poisson model only needs (n_events, total_hours) per split
for fitting, so this loader is intentionally thin — no gap sequence, no
covariates. (models/poisson/evaluate.py, which needs the raw per-event
gaps for scoring, calls core.splits.get_chronological_windows directly
rather than going through this class.)
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from arrivalcast.core.splits import get_chronological_windows


@dataclass
class PoissonSplit:
    n_events: int
    total_hours: float
    start: pd.Timestamp
    end: pd.Timestamp


class PoissonDataLoader:
    """
    Prepares train/val/test statistics for HomogeneousPoisson from one
    source's raw timestamps.
    """

    def __init__(
        self,
        timestamps: pd.Series,
        train_frac: float = 0.7,
        val_frac: float = 0.15,
        global_study_end: pd.Timestamp | None = None,
    ):
        """
        global_study_end : the real end of the observation period (dataset
            generation cutoff / extraction date). Used as the bound for the
            LAST split only — see core.splits.get_chronological_windows
            for the full explanation (shared by every model's dataloader).
        """
        self.timestamps = pd.Series(timestamps).sort_values().reset_index(drop=True)
        self.train_frac = train_frac
        self.val_frac = val_frac
        self.global_study_end = global_study_end

    def get_splits(self) -> dict[str, PoissonSplit]:
        splits_raw = get_chronological_windows(
            self.timestamps, self.train_frac, self.val_frac, self.global_study_end
        )
        splits = {}
        for name, sw in splits_raw.items():
            if sw is None:
                splits[name] = PoissonSplit(0, 0.0, pd.NaT, pd.NaT)
                continue
            window = sw.window
            splits[name] = PoissonSplit(
                n_events=window.n_events,
                total_hours=max(window.total_hours, 1e-6),
                start=window.t0,
                end=window.study_end,
            )
        return splits
