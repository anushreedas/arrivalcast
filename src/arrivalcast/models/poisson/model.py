"""
Poisson process baseline.

Two variants are provided:

- HomogeneousPoisson: constant rate lambda. Sufficient statistics are just
  (n_events, total_hours) — this is the simplest possible baseline and has
  NO slot for calendar covariates.

- PiecewiseConstantPoisson: rate depends on a discrete calendar bucket
  (e.g. hour-of-day x weekday/weekend). This is the one legitimate way to
  fold calendar structure into a Poisson model without changing its
  functional form to something else — see the module-level note in
  features/calendar_features.py. It's included as an optional
  extension/comparison point, not the primary baseline.

Neither variant has any history-dependence (no self-excitation); that's
what distinguishes this from Hawkes.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class HomogeneousPoisson:
    rate_per_hour: float | None = None

    def fit(self, n_events: int, total_hours: float) -> HomogeneousPoisson:
        if total_hours <= 0:
            raise ValueError("total_hours must be > 0")
        self.rate_per_hour = n_events / total_hours
        return self

    def log_likelihood(self, n_events: int, total_hours: float) -> float:
        """Poisson process log-likelihood: n*log(lambda) - lambda*T."""
        if self.rate_per_hour is None:
            raise RuntimeError("call .fit() first")
        lam = self.rate_per_hour
        if lam <= 0:
            return -np.inf
        return n_events * np.log(lam) - lam * total_hours

    def predict_next_gap_hours(self) -> float:
        """Expected time to next event: 1/lambda (memoryless, so this is
        independent of how long it's already been since the last event)."""
        if self.rate_per_hour is None:
            raise RuntimeError("call .fit() first")
        return 1.0 / self.rate_per_hour


class PiecewiseConstantPoisson:
    """
    Inhomogeneous Poisson with a rate that's constant within each calendar
    bucket (e.g. hour-of-day, or hour-of-day x is_weekend). Rate for each
    bucket is estimated as (events observed in that bucket) / (total hours
    of exposure to that bucket) — the standard piecewise-constant-hazard MLE.
    """

    def __init__(self, bucket_fn):
        """
        bucket_fn : callable(pd.Series[datetime]) -> pd.Series[hashable]
            Maps timestamps to a discrete bucket label, e.g.
            `lambda ts: ts.dt.hour` for hour-of-day buckets.
        """
        self.bucket_fn = bucket_fn
        self.rates_: dict | None = None

    def fit(
        self,
        event_timestamps: pd.Series,
        t0: pd.Timestamp,
        study_end: pd.Timestamp,
        freq: str = "h",
    ) -> PiecewiseConstantPoisson:
        """
        Estimate one rate per bucket by counting events per bucket and
        dividing by total exposure hours per bucket, where exposure is
        computed by tiling the observation window at `freq` resolution.
        """
        exposure_grid = pd.date_range(t0, study_end, freq=freq)
        exposure_buckets = self.bucket_fn(pd.Series(exposure_grid).dt)
        exposure_hours = (
            pd.Series(exposure_buckets).value_counts()
            * pd.Timedelta(freq).total_seconds()
            / 3600.0
        )

        event_buckets = self.bucket_fn(pd.to_datetime(event_timestamps).dt)
        event_counts = pd.Series(event_buckets).value_counts()

        self.rates_ = {}
        for bucket, hours in exposure_hours.items():
            count = event_counts.get(bucket, 0)
            self.rates_[bucket] = count / hours if hours > 0 else 0.0
        return self

    def rate_at(self, timestamp: pd.Timestamp) -> float:
        if self.rates_ is None:
            raise RuntimeError("call .fit() first")
        bucket = self.bucket_fn(pd.Series([timestamp]).dt).iloc[0]
        return self.rates_.get(bucket, 0.0)
