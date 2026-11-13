from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

torch.set_num_threads(1)


@pytest.fixture
def daily_timestamps() -> pd.Series:
    """~120 daily arrivals near 09:00 with small jitter (a 'regular cadence' source)."""
    rng = np.random.default_rng(0)
    base = pd.date_range("2023-01-01 09:00", periods=120, freq="D")
    jitter = pd.to_timedelta(rng.normal(0, 15, size=len(base)), unit="min")
    return pd.Series(base + jitter)


@pytest.fixture
def tiny_dataset(tmp_path, daily_timestamps):
    """Two sources in the project's 3-column schema, written as one CSV per source."""
    d = tmp_path / "synthetic"
    d.mkdir()
    for sid, ts in (("SRC_A", daily_timestamps),
                    ("SRC_B", pd.Series(pd.date_range("2023-01-02 07:00", periods=40, freq="7D")))):
        pd.DataFrame({"source_id": sid, "pattern_description": f"test {sid}",
                      "arrival_timestamp": ts}).to_csv(d / f"{sid}.csv", index=False)
    return d
