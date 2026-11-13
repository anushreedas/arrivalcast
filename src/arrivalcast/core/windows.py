"""
Observation-window bookkeeping shared by every model.

Every temporal point process model needs three things beyond the raw
timestamps: the inter-arrival gaps, a numeric time origin, and the length of
the *unobserved tail* after the last event (used in the survival/compensator
term of the log-likelihood). Getting the window boundary wrong is one of the
easiest ways to silently bias every model in the same direction, so it's
centralized here rather than re-derived per model.

IMPORTANT: the observation-window end T should be the end of the observation
period (e.g. "the day this dataset was extracted"), NOT the last arrival
timestamp. If you set T = last arrival, every sequence's "gap after the last
event" is exactly zero, which erases the very censoring signal the survival
term is supposed to capture. Pass `study_end` explicitly wherever it's known;
the fallback (last arrival) is only a reasonable stand-in when a real
end-of-study boundary isn't available (e.g. in this project's synthetic
data, generated over a fixed 2023-01-01..2024-12-31 window).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class ObservationWindow:
    """
    t0, study_end : pandas.Timestamp
        Start and end of the observation period.
    gaps_hours : np.ndarray, shape (n_events - 1,)
        Inter-arrival gaps between consecutive events, in hours.
    first_gap_hours : float
        Elapsed time from t0 to the first observed event, in hours.
    tail_hours : float
        Elapsed time from the last observed event to study_end, in hours
        (the right-censored / survival interval).
    n_events : int
    """

    t0: pd.Timestamp
    study_end: pd.Timestamp
    gaps_hours: np.ndarray
    first_gap_hours: float
    tail_hours: float
    n_events: int

    @property
    def total_hours(self) -> float:
        return (self.study_end - self.t0).total_seconds() / 3600.0


def compute_window(
    timestamps: pd.Series,
    study_start: pd.Timestamp | None = None,
    study_end: pd.Timestamp | None = None,
) -> ObservationWindow:
    """
    Compute inter-arrival gaps and window bounds for one source.

    Parameters
    ----------
    timestamps : pd.Series of datetime64, already sorted ascending.
    study_start : optional explicit t0. Defaults to the first arrival.
    study_end : optional explicit end of observation. Defaults to the last
        arrival — see the module docstring caveat above.
    """
    ts = pd.Series(timestamps).sort_values().reset_index(drop=True)
    if len(ts) == 0:
        raise ValueError("timestamps is empty")

    t0 = study_start if study_start is not None else ts.iloc[0]
    end = study_end if study_end is not None else ts.iloc[-1]

    gaps_hours = ts.diff().dropna().dt.total_seconds().to_numpy() / 3600.0
    first_gap_hours = (ts.iloc[0] - t0).total_seconds() / 3600.0
    tail_hours = max((end - ts.iloc[-1]).total_seconds() / 3600.0, 0.0)

    return ObservationWindow(
        t0=t0,
        study_end=end,
        gaps_hours=gaps_hours,
        first_gap_hours=first_gap_hours,
        tail_hours=tail_hours,
        n_events=len(ts),
    )
