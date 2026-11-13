"""
Loading raw arrival data into a common in-memory representation.

Both datasets (synthetic and BTS) are produced by the pipeline with the same
3-column schema: source_id, pattern_description, arrival_timestamp. This
module is the ONLY place that should know about file paths / formats — every
downstream consumer (feature helpers, model dataloaders) works off the
dict-of-DataFrames shape returned here, regardless of whether the data lived
in one combined file or one file per source.

Supports two on-disk layouts, since your local synthetic data is one CSV per
source (data/synthetic/SRC_01.csv, SRC_02.csv, ...) while this repo's sample
data is a single combined CSV — both are handled transparently:

    - a directory containing one CSV per source
    - a single CSV/Parquet file containing multiple source_id values
"""
from __future__ import annotations

import glob
import os
from typing import Dict

import pandas as pd

SOURCE_COL = "source_id"
PATTERN_COL = "pattern_description"
TS_COL = "arrival_timestamp"
REQUIRED_COLS = (SOURCE_COL, PATTERN_COL, TS_COL)


def _read_any(path: str) -> pd.DataFrame:
    if path.endswith(".parquet"):
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    return df


def _load_path(path: str) -> pd.DataFrame:
    """Read a single combined file, or concatenate every file in a directory."""
    if os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, "*.csv"))) + sorted(
            glob.glob(os.path.join(path, "*.parquet"))
        )
        if not files:
            raise FileNotFoundError(f"No .csv/.parquet files found under {path}")
        frames = [_read_any(f) for f in files]
        return pd.concat(frames, ignore_index=True)
    return _read_any(path)


def load_sources(path: str) -> Dict[str, pd.DataFrame]:
    """
    Load raw arrivals and split into one DataFrame per source_id.

    Each returned DataFrame has a single column, `arrival_timestamp`
    (parsed to pandas datetime, sorted ascending, index reset), and carries
    the source's pattern_description as a `.attrs["pattern_description"]`
    string for convenience in plots/reports.

    Parameters
    ----------
    path : str
        Either a directory of per-source files or one combined file.

    Returns
    -------
    dict[str, pd.DataFrame]
    """
    raw = _load_path(path)
    # Always end up with tz-naive UTC wall-clock times. The synthetic data is
    # naive; a BTS parquet may be tz-aware (the data description says UTC), and
    # mixing aware and naive timestamps breaks the study-end arithmetic.
    raw[TS_COL] = pd.to_datetime(raw[TS_COL], utc=True, format="ISO8601").dt.tz_localize(None)

    out: Dict[str, pd.DataFrame] = {}
    for source_id, group in raw.groupby(SOURCE_COL):
        g = group.sort_values(TS_COL).reset_index(drop=True)
        pattern = g[PATTERN_COL].iloc[0] if PATTERN_COL in g.columns else ""
        series = g[[TS_COL]].copy()
        series.attrs["pattern_description"] = pattern
        series.attrs["source_id"] = source_id
        out[source_id] = series
    return out


def load_synthetic(path: str = "data/synthetic") -> Dict[str, pd.DataFrame]:
    """Convenience wrapper — see load_sources()."""
    return load_sources(path)


def load_bts(path: str = "data/bts/DL_JFK-LAX.parquet") -> Dict[str, pd.DataFrame]:
    """Convenience wrapper — see load_sources()."""
    return load_sources(path)


def load_all_sources(
    synthetic_path: str = "data/synthetic",
    bts_path: str = "data/bts/DL_JFK-LAX.parquet",
) -> Dict[str, pd.DataFrame]:
    """
    Load both datasets into a single dict keyed by source_id.

    Every entry also gets `.attrs["dataset"]` set to "synthetic" or "bts" so
    downstream code (e.g. the comparison notebook) can group/filter by
    dataset without re-parsing the source_id naming convention.
    """
    combined: Dict[str, pd.DataFrame] = {}

    if synthetic_path and os.path.exists(synthetic_path):
        for source_id, df in load_synthetic(synthetic_path).items():
            df.attrs["dataset"] = "synthetic"
            combined[source_id] = df

    if bts_path and os.path.exists(bts_path):
        for source_id, df in load_bts(bts_path).items():
            df.attrs["dataset"] = "bts"
            combined[source_id] = df

    return combined
