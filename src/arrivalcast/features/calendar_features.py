"""
Calendar/covariate feature extraction from raw arrival timestamps.

These features are only meaningful to models whose functional form has a
slot for covariates — that's the neural models (Neural Hawkes, LogNormMix),
or a hand-built inhomogeneous-rate extension of Poisson/Hawkes. The plain
homogeneous Poisson and classical exponential-kernel Hawkes baselines in
this project have no such slot, so this module is consumed by
models/neural_hawkes and models/lognormmix only (see models/poisson and
models/hawkes docstrings for why calendar features don't apply there).

All features are computed directly from `arrival_timestamp` — nothing here
requires data that wouldn't be available in real time as a new timestamp
comes in, which matters since the production system only ever sees raw
timestamps and must derive features on the fly.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

try:
    from pandas.tseries.holiday import USFederalHolidayCalendar

    _HOLIDAY_CAL = USFederalHolidayCalendar()
except ImportError:  # pragma: no cover
    _HOLIDAY_CAL = None


def compute_calendar_features(timestamps: pd.Series) -> pd.DataFrame:
    """
    Derive calendar covariates for a sequence of arrival timestamps.

    Cyclical fields (hour-of-day, day-of-week, month) are encoded as
    sin/cos pairs rather than raw integers or one-hot, so that e.g. 23:00
    and 00:00 are close in feature space instead of maximally far apart —
    this matters for the neural models, which otherwise have no reason to
    know that hour 23 and hour 0 are adjacent.

    Parameters
    ----------
    timestamps : pd.Series of datetime64

    Returns
    -------
    pd.DataFrame, same length/order as `timestamps`, columns:
        hour_sin, hour_cos       - hour of day (0-23), cyclical
        dow_sin, dow_cos         - day of week (0=Mon..6=Sun), cyclical
        month_sin, month_cos     - month (1-12), cyclical
        is_weekend               - 1.0 if Sat/Sun else 0.0
        is_business_hours        - 1.0 if 09:00-17:00 local (as given) else 0.0
        is_us_federal_holiday    - 1.0 if the calendar date is a US federal
                                    holiday else 0.0 (0.0 if the holiday
                                    calendar is unavailable)
    """
    ts = pd.to_datetime(pd.Series(timestamps)).reset_index(drop=True)

    hour = ts.dt.hour + ts.dt.minute / 60.0
    dow = ts.dt.dayofweek  # Monday=0
    month = ts.dt.month

    feats = pd.DataFrame(
        {
            "hour_sin": np.sin(2 * np.pi * hour / 24.0),
            "hour_cos": np.cos(2 * np.pi * hour / 24.0),
            "dow_sin": np.sin(2 * np.pi * dow / 7.0),
            "dow_cos": np.cos(2 * np.pi * dow / 7.0),
            "month_sin": np.sin(2 * np.pi * (month - 1) / 12.0),
            "month_cos": np.cos(2 * np.pi * (month - 1) / 12.0),
            "is_weekend": (dow >= 5).astype(float),
            "is_business_hours": ((ts.dt.hour >= 9) & (ts.dt.hour < 17)).astype(float),
        }
    )

    if _HOLIDAY_CAL is not None and len(ts) > 0:
        holidays = _HOLIDAY_CAL.holidays(start=ts.min().normalize(), end=ts.max().normalize())
        feats["is_us_federal_holiday"] = ts.dt.normalize().isin(holidays).astype(float)
    else:
        feats["is_us_federal_holiday"] = 0.0

    return feats


CALENDAR_FEATURE_NAMES = [
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "month_sin",
    "month_cos",
    "is_weekend",
    "is_business_hours",
    "is_us_federal_holiday",
]
