"""
arrivalcast.data.pipeline
================
CLI tool for generating the file-arrival datasets.

This script produces arrival timestamps and calendar facts (day of week,
hour, holiday flag) with a fixed, uniform schema across every source.

Two independent subcommands, each capable of producing output for a
single source in one run, or for every registered source in one run:

  synthetic   Generate one (or all) of the 26 synthetic ingestion
              sources over a configurable date range. Each source's
              randomness is seeded deterministically from its own
              source id, so generating a single source on its own
              always reproduces exactly the same data as generating
              it as part of a full batch run.

  bts         Extract one (or all) BTS Airline On-Time Performance
              "routes" (origin-destination-carrier) from locally
              staged monthly ZIP files (or by downloading them) into
              a clean, analysis-ready table.

Examples
--------
    # List available synthetic source patterns
    arrivalcast data synthetic --list

    # Generate a single source
    arrivalcast data synthetic --source SRC_24 \\
        --start 2023-01-01 --end 2024-12-31 --output-dir data/synthetic

    # Generate every synthetic source (one file per source, plus a
    # combined file covering all sources)
    arrivalcast data synthetic --source all --output-dir data/synthetic

    # List the predefined BTS routes
    arrivalcast data bts --list

    # Extract a single BTS route from locally staged monthly ZIPs
    arrivalcast data bts --route JFK-LAX-DL \\
        --years 2023 2024 --raw-dir data/airline_flights_raw \\
        --output-dir data/bts

    # Extract every predefined BTS route, downloading months on the fly
    arrivalcast data bts --route all --mode download \\
        --years 2023 2024 --output-dir data/bts

Dependencies
------------
    pip install pandas numpy requests tqdm holidays pyarrow
"""

from __future__ import annotations

import argparse
import hashlib
import io
import logging
import sys
import zipfile
from calendar import monthrange
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import holidays
import numpy as np
import pandas as pd
import requests
from scipy.stats import pareto
from tqdm import tqdm

logger = logging.getLogger("data_pipeline")

# ═════════════════════════════════════════════════════════════════════════
# Global constants
# ═════════════════════════════════════════════════════════════════════════
BASE_SEED = 42
US_HOLIDAYS = set(holidays.US(years=list(range(2022, 2027))).keys())

BTS_URL = (
    "https://transtats.bts.gov/PREZIP/"
    "On_Time_Reporting_Carrier_On_Time_Performance_1987_present_{year}_{month}.zip"
)
BTS_HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Accept": "application/zip,*/*",
    "Referer": "https://transtats.bts.gov/",
}
BTS_KEEP_COLS = [
    "Year", "Month", "DayofMonth", "DayOfWeek",
    "Reporting_Airline", "Flight_Number_Reporting_Airline", "Origin", "Dest",
    "CRSDepTime", "DepTime", "DepDelay",
    "CRSArrTime", "ArrTime", "ArrDelay",
    "Cancelled", "CancellationCode", "Diverted", "Distance",
]


# ═════════════════════════════════════════════════════════════════════════
# Logging
# ═════════════════════════════════════════════════════════════════════════
def configure_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
    )


# ═════════════════════════════════════════════════════════════════════════
# Deterministic, per-source random number generation
# ═════════════════════════════════════════════════════════════════════════
def make_rng(key: str, base_seed: int = BASE_SEED) -> np.random.Generator:
    """
    Build a reproducible RNG for `key` (a source id or BTS route key) that
    is independent of what else is generated in the same process. This is
    what allows a single source to be regenerated in isolation and get
    byte-for-byte identical output to a full batch run.
    """
    digest = hashlib.sha256(f"{base_seed}:{key}".encode()).hexdigest()
    seed = int(digest[:16], 16) % (2**32)
    return np.random.default_rng(seed)


# ═════════════════════════════════════════════════════════════════════════
# Calendar helpers (pure functions, no randomness)
# ═════════════════════════════════════════════════════════════════════════
def is_holiday(d) -> bool:
    if isinstance(d, datetime):
        return d.date() in US_HOLIDAYS
    return d in US_HOLIDAYS


def is_weekday(d: datetime) -> bool:
    return d.weekday() < 5


def first_business_day(year: int, month: int) -> datetime:
    d = datetime(year, month, 1)
    while d.weekday() >= 5 or is_holiday(d):
        d += timedelta(days=1)
    return d


def last_business_day(year: int, month: int) -> datetime:
    last = monthrange(year, month)[1]
    d = datetime(year, month, last)
    while d.weekday() >= 5 or is_holiday(d):
        d -= timedelta(days=1)
    return d


def nth_weekday_of_month(year: int, month: int, weekday: int, n: int) -> Optional[datetime]:
    """Return the nth occurrence of weekday (0=Mon…6=Sun) in year/month."""
    count = 0
    d = datetime(year, month, 1)
    while d.month == month:
        if d.weekday() == weekday:
            count += 1
            if count == n:
                return d
        d += timedelta(days=1)
    return None


def is_end_of_quarter(d: datetime) -> bool:
    quarter_ends = {
        (3, 29), (3, 30), (3, 31),
        (6, 28), (6, 29), (6, 30),
        (9, 28), (9, 29), (9, 30),
        (12, 29), (12, 30), (12, 31),
    }
    return (d.month, d.day) in quarter_ends


def all_days(start: datetime, end: datetime):
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)


def parse_date(value: str) -> datetime:
    try:
        return datetime.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid date {value!r}; expected YYYY-MM-DD"
        ) from exc


# ═════════════════════════════════════════════════════════════════════════
# Randomness helpers (RNG passed explicitly — never a module-level global)
# ═════════════════════════════════════════════════════════════════════════
def gauss_jitter(rng: np.random.Generator, sd_hours: float = 0.25) -> timedelta:
    return timedelta(hours=float(rng.normal(0, sd_hours)))


def uniform_jitter(rng: np.random.Generator, low_min: float = -5, high_min: float = 5) -> timedelta:
    return timedelta(minutes=float(rng.uniform(low_min, high_min)))


def at(rng: np.random.Generator, d: datetime, hour: float, jitter_sd: float = 0.25) -> datetime:
    """Scheduled arrival at fractional UTC hour on date d, plus Gaussian noise."""
    h = int(hour)
    m = int((hour - h) * 60)
    return datetime(d.year, d.month, d.day, h, m) + gauss_jitter(rng, jitter_sd)


def rec(src_id: str, description: str, arrival: datetime) -> dict:
    """
    Build a standard arrival record. Every source produces exactly this
    fixed schema — no source-specific bonus columns — so that downstream
    processing (and any future production automation) can treat all
    sources identically without per-source special-casing.
    """
    return {
        "source_id": src_id,
        "pattern_description": description,
        "arrival_timestamp": arrival
    }


# ═════════════════════════════════════════════════════════════════════════
# PART 1 — Synthetic source generators
# Every generator has the signature (start, end, rng) -> List[dict] and is
# registered below with a stable source_id and a human-readable description.
# ═════════════════════════════════════════════════════════════════════════
GeneratorFn = Callable[[datetime, datetime, np.random.Generator], List[dict]]


@dataclass(frozen=True)
class SourceSpec:
    source_id: str
    description: str
    generator: GeneratorFn


SYNTHETIC_REGISTRY: Dict[str, SourceSpec] = {}


def _register(source_id: str, description: str, fn: GeneratorFn) -> None:
    if source_id in SYNTHETIC_REGISTRY:
        raise ValueError(f"Duplicate source_id registered: {source_id}")
    SYNTHETIC_REGISTRY[source_id] = SourceSpec(source_id, description, fn)


def gen_src01(s, e, rng):
    return [rec("SRC_01", SYNTHETIC_REGISTRY["SRC_01"].description, at(rng, day, 9))
            for day in all_days(s, e)]


def gen_src02(s, e, rng):
    desc = SYNTHETIC_REGISTRY["SRC_02"].description
    out = []
    for day in all_days(s, e):
        h = 13 if day.weekday() == 2 else 9
        out.append(rec("SRC_02", desc, at(rng, day, h)))
    return out


def gen_src03(s, e, rng):
    desc = SYNTHETIC_REGISTRY["SRC_03"].description
    return [rec("SRC_03", desc, at(rng, day, 14)) for day in all_days(s, e) if is_weekday(day)]


def gen_src04(s, e, rng):
    desc = SYNTHETIC_REGISTRY["SRC_04"].description
    out, y, m = [], s.year, s.month
    while True:
        fbd = first_business_day(y, m)
        if fbd > e:
            break
        out.append(rec("SRC_04", desc, at(rng, fbd, 6, jitter_sd=0.5)))
        m += 1
        if m > 12:
            m, y = 1, y + 1
    return out


def gen_src05(s, e, rng):
    desc = SYNTHETIC_REGISTRY["SRC_05"].description
    out = []
    for yr in range(s.year, e.year + 1):
        for mo in (1, 4, 7, 10):
            fbd = first_business_day(yr, mo)
            if s <= fbd <= e:
                out.append(rec("SRC_05", desc, at(rng, fbd, 6, jitter_sd=0.5)))
    return out


def gen_src06(s, e, rng):
    desc = SYNTHETIC_REGISTRY["SRC_06"].description
    return [rec("SRC_06", desc, at(rng, day, 11))
            for day in all_days(s, e) if day.weekday() in (1, 2, 3)]


def gen_src07(s, e, rng):
    desc = SYNTHETIC_REGISTRY["SRC_07"].description
    return [rec("SRC_07", desc, at(rng, day, 8))
            for day in all_days(s, e) if is_weekday(day) and not is_holiday(day)]


def gen_src08(s, e, rng):
    """One arrival per hour every hour, +/- 3 min uniform noise."""
    desc = SYNTHETIC_REGISTRY["SRC_08"].description
    out = []
    for day in all_days(s, e):
        for h in range(24):
            arrival = datetime(day.year, day.month, day.day, h, 0) + uniform_jitter(rng, -3, 3)
            out.append(rec("SRC_08", desc, arrival))
    return out


def gen_src09(s, e, rng):
    desc = SYNTHETIC_REGISTRY["SRC_09"].description
    out = []
    for day in all_days(s, e):
        for h in (0, 6, 12, 18):
            out.append(rec("SRC_09", desc, at(rng, day, h, jitter_sd=0.1)))
    return out


def gen_src10(s, e, rng):
    desc = SYNTHETIC_REGISTRY["SRC_10"].description
    out = []
    for day in all_days(s, e):
        if is_weekday(day):
            for h in (8, 18):
                out.append(rec("SRC_10", desc, at(rng, day, h)))
    return out


def gen_src11(s, e, rng):
    desc = SYNTHETIC_REGISTRY["SRC_11"].description
    out = []
    for day in all_days(s, e):
        if is_weekday(day):
            for h in (8, 10, 12, 14, 16):
                out.append(rec("SRC_11", desc, at(rng, day, h)))
    return out


def gen_src12(s, e, rng):
    desc = SYNTHETIC_REGISTRY["SRC_12"].description
    return [rec("SRC_12", desc, at(rng, day, 7)) for day in all_days(s, e) if day.weekday() == 0]


def gen_src13(s, e, rng):
    desc = SYNTHETIC_REGISTRY["SRC_13"].description
    return [rec("SRC_13", desc, at(rng, day, 16, jitter_sd=2.0))
            for day in all_days(s, e) if day.weekday() == 4]


def gen_src14(s, e, rng):
    desc = SYNTHETIC_REGISTRY["SRC_14"].description
    out = []
    for day in all_days(s, e):
        h = 9 if is_weekday(day) else 15
        out.append(rec("SRC_14", desc, at(rng, day, h)))
    return out


def gen_src15(s, e, rng):
    desc = SYNTHETIC_REGISTRY["SRC_15"].description
    mondays = [day for day in all_days(s, e) if day.weekday() == 0]
    return [rec("SRC_15", desc, at(rng, day, 7)) for i, day in enumerate(mondays) if i % 2 == 0]


def gen_src16(s, e, rng):
    desc = SYNTHETIC_REGISTRY["SRC_16"].description
    out, y, m = [], s.year, s.month
    while True:
        lbd = last_business_day(y, m)
        if lbd > e:
            break
        if lbd >= s:
            out.append(rec("SRC_16", desc, at(rng, lbd, 16, jitter_sd=1.0)))
        m += 1
        if m > 12:
            m, y = 1, y + 1
    return out


def gen_src17(s, e, rng):
    desc = SYNTHETIC_REGISTRY["SRC_17"].description
    out = []
    for yr in range(s.year, e.year + 1):
        for mo in range(1, 13):
            for day_target in (5, 15, 25):
                try:
                    day = datetime(yr, mo, day_target)
                except ValueError:
                    continue
                while day.weekday() >= 5 or is_holiday(day):
                    day += timedelta(days=1)
                if s <= day <= e:
                    out.append(rec("SRC_17", desc, at(rng, day, 9)))
    return out


def gen_src18(s, e, rng):
    desc = SYNTHETIC_REGISTRY["SRC_18"].description
    out = []
    for yr in range(s.year, e.year + 1):
        for mo in range(1, 13):
            day = nth_weekday_of_month(yr, mo, weekday=1, n=2)  # Tue=1
            if day and s <= day <= e:
                out.append(rec("SRC_18", desc, at(rng, day, 10)))
    return out


def gen_src19(s, e, rng):
    """Daily at ~10:00 UTC; each day independently has a 5% outage probability."""
    desc = SYNTHETIC_REGISTRY["SRC_19"].description
    out = []
    for day in all_days(s, e):
        if rng.random() < 0.05:
            continue
        out.append(rec("SRC_19", desc, at(rng, day, 10)))
    return out


def gen_src20(s, e, rng):
    """Daily at ~10:00 UTC; timing noise sd is 3x larger in Dec/Jan."""
    desc = SYNTHETIC_REGISTRY["SRC_20"].description
    out = []
    for day in all_days(s, e):
        sd = 0.75 if day.month in (12, 1) else 0.25
        out.append(rec("SRC_20", desc, at(rng, day, 10, jitter_sd=sd)))
    return out


def gen_src21(s, e, rng):
    """Daily at ~09:00 UTC; 3 files/day on the last 3 days of each quarter."""
    desc = SYNTHETIC_REGISTRY["SRC_21"].description
    out = []
    for day in all_days(s, e):
        if is_end_of_quarter(day):
            for h in (8, 13, 17):
                out.append(rec("SRC_21", desc, at(rng, day, h)))
        else:
            out.append(rec("SRC_21", desc, at(rng, day, 9)))
    return out


def gen_src22(s, e, rng):
    """Weekdays at ~09:00 UTC; 8% chance of a correction file ~3h later."""
    desc = SYNTHETIC_REGISTRY["SRC_22"].description
    out = []
    for day in all_days(s, e):
        if not is_weekday(day):
            continue
        primary = at(rng, day, 9)
        out.append(rec("SRC_22", desc, primary))
        if rng.random() < 0.08:
            correction = primary + timedelta(hours=3) + gauss_jitter(rng, 0.2)
            out.append(rec("SRC_22", desc, correction))
    return out


def gen_src23(s, e, rng):
    desc = SYNTHETIC_REGISTRY["SRC_23"].description
    out = []
    for yr in range(s.year, e.year + 1):
        fbd = first_business_day(yr, 1)
        if s <= fbd <= e:
            out.append(rec("SRC_23", desc, at(rng, fbd, 6, jitter_sd=0.5)))
    return out


def gen_src24(s, e, rng):
    """
    Self-exciting (Hawkes-like): one background event/day at a random time
    in [06:00, 20:00) UTC. Each event independently spawns 0 or 1 offspring
    with probability alpha=0.4, after an Exponential(mean=1/beta=2h) delay
    capped at 4h. Mimics a univariate Hawkes process with exponential decay:
        lambda*(t) = mu + alpha*beta * sum_{t_i<t} exp(-beta*(t - t_i))
    """
    desc = SYNTHETIC_REGISTRY["SRC_24"].description
    alpha, beta, max_delay = 0.4, 0.5, 4.0
    out = []
    for day in all_days(s, e):
        base_hour = 6.0 + rng.uniform(0, 14)
        h = int(base_hour)
        m = int((base_hour - h) * 60)
        parent = datetime(day.year, day.month, day.day, h, m) + uniform_jitter(rng, -2, 2)
        out.append(rec("SRC_24", desc, parent))
        if rng.random() < alpha:
            delay_hrs = min(float(rng.exponential(1.0 / beta)), max_delay)
            offspring = parent + timedelta(hours=delay_hrs)
            if offspring.date() <= e.date():
                out.append(rec("SRC_24", desc, offspring))
    return out


def gen_src25(s, e, rng):
    """
    Inter-arrival times ~ Pareto(b=1.5, scale=3h); mean gap = 9h. Shape
    b=1.5 gives finite mean but infinite variance -> heavy-tailed gaps.
    """
    desc = SYNTHETIC_REGISTRY["SRC_25"].description
    b_shape, scale = 1.5, 3.0
    out = []
    current = datetime(s.year, s.month, s.day, 0, 0)
    while current <= e:
        out.append(rec("SRC_25", desc, current))
        gap_hrs = float(pareto.rvs(b_shape, scale=scale, random_state=rng))
        current += timedelta(hours=gap_hrs)
    return out


def gen_src26(s, e, rng):
    """
    Weekdays, base time 09:00 UTC. Delay (min) = 0.5 * previous_file_size_mb,
    capped at 120 min. File sizes ~ LogNormal(mu=3, sigma=1), median ~20MB.
    File size itself is not recorded in the output — only its effect on
    arrival timing — so this source's delay driver is latent from the
    model's point of view (see note in module docstring).
    """
    desc = SYNTHETIC_REGISTRY["SRC_26"].description
    delay_per_mb, max_delay = 0.5, 120.0
    out = []
    prev_size_mb = 20.0
    for day in all_days(s, e):
        if not is_weekday(day):
            continue
        delay_min = min(delay_per_mb * prev_size_mb, max_delay)
        arrival = (datetime(day.year, day.month, day.day, 9, 0)
                   + timedelta(minutes=delay_min) + gauss_jitter(rng, 0.1))
        size_mb = float(rng.lognormal(mean=3.0, sigma=1.0))
        out.append(rec("SRC_26", desc, arrival))
        prev_size_mb = size_mb
    return out


_register("SRC_01", "Daily at ~09:00 UTC — every calendar day", gen_src01)
_register("SRC_02", "Daily at ~09:00 UTC, ~4h late on Wednesdays", gen_src02)
_register("SRC_03", "Weekdays only at ~14:00 UTC", gen_src03)
_register("SRC_04", "Monthly — first business day at ~06:00 UTC", gen_src04)
_register("SRC_05", "Quarterly — first biz day of Jan/Apr/Jul/Oct at ~06:00 UTC", gen_src05)
_register("SRC_06", "Tuesday, Wednesday, Thursday only at ~11:00 UTC", gen_src06)
_register("SRC_07", "Weekdays excluding US federal holidays at ~08:00 UTC", gen_src07)
_register("SRC_08", "Hourly 24/7", gen_src08)
_register("SRC_09", "Every 6 hours (00/06/12/18 UTC), every day", gen_src09)
_register("SRC_10", "Twice daily on weekdays (~08:00 and ~18:00 UTC)", gen_src10)
_register("SRC_11", "Business hours every 2h on weekdays (08/10/12/14/16 UTC)", gen_src11)
_register("SRC_12", "Weekly — Mondays only at ~07:00 UTC", gen_src12)
_register("SRC_13", "Weekly — Fridays at ~16:00 UTC, high jitter (+/-2h)", gen_src13)
_register("SRC_14", "Weekdays at ~09:00 UTC + reduced weekend service at ~15:00 UTC", gen_src14)
_register("SRC_15", "Bi-weekly — alternating Mondays at ~07:00 UTC", gen_src15)
_register("SRC_16", "Last business day of each month at ~16:00 UTC (higher jitter)", gen_src16)
_register("SRC_17", "5th/15th/25th of month (or next biz day) at ~09:00 UTC", gen_src17)
_register("SRC_18", "Second Tuesday of each month at ~10:00 UTC", gen_src18)
_register("SRC_19", "Daily at ~10:00 UTC with 5% random outage probability", gen_src19)
_register("SRC_20", "Daily at ~10:00 UTC — 3x jitter in Dec and Jan", gen_src20)
_register("SRC_21", "Daily at ~09:00 UTC + 3 files/day on last 3 days of quarter", gen_src21)
_register("SRC_22", "Weekdays at ~09:00 UTC; 8% chance of correction file ~3h later", gen_src22)
_register("SRC_23", "Annual — first business day of the year at ~06:00 UTC", gen_src23)
_register("SRC_24", "Self-exciting (Hawkes-like): daily background + offspring within 4h "
                     "(alpha=0.4, beta=0.5/h)", gen_src24)
_register("SRC_25", "Heavy-tail inter-arrivals: Pareto(b=1.5, scale=3h)", gen_src25)
_register("SRC_26", "Weekdays at ~09:00 UTC; delay proportional to previous file size "
                     "(LogNormal sizes, 0.5 min/MB, cap 120 min)", gen_src26)


# ═════════════════════════════════════════════════════════════════════════
# Synthetic generation orchestration
# ═════════════════════════════════════════════════════════════════════════
def generate_synthetic_source(
        source_id: str, start: datetime, end: datetime, base_seed: int = BASE_SEED
) -> pd.DataFrame:
    """Generate one source's arrival history as a fully-featured DataFrame."""
    if source_id not in SYNTHETIC_REGISTRY:
        raise KeyError(
            f"Unknown source_id {source_id!r}. Run with --list to see valid ids."
        )
    spec = SYNTHETIC_REGISTRY[source_id]
    rng = make_rng(source_id, base_seed)
    records = spec.generator(start, end, rng)
    if not records:
        logger.warning("Source %s produced zero arrivals in [%s, %s]", source_id, start, end)
        return pd.DataFrame()

    df = pd.DataFrame(records)
    df["arrival_timestamp"] = pd.to_datetime(df["arrival_timestamp"])
    df = df.sort_values("arrival_timestamp").reset_index(drop=True)
    return df


def save_dataframe_outputs(df: pd.DataFrame, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    data_path = output_dir / f"{stem}.csv"
    df.to_csv(data_path, index=False)
    logger.info("Saved %s (%d rows)", data_path, len(df))


def run_synthetic_single(source_id: str, start: datetime, end: datetime,
                         output_dir: Path, base_seed: int) -> bool:
    logger.info("Generating synthetic source %s [%s -> %s]", source_id, start.date(), end.date())
    df = generate_synthetic_source(source_id, start, end, base_seed)
    if df.empty:
        logger.error("No data generated for %s — nothing written.", source_id)
        return False
    save_dataframe_outputs(df, output_dir, source_id)
    return True


def run_synthetic_all(start: datetime, end: datetime, output_dir: Path, base_seed: int) -> bool:
    logger.info("Generating all %d synthetic sources [%s -> %s]",
                len(SYNTHETIC_REGISTRY), start.date(), end.date())
    frames = []
    for source_id in sorted(SYNTHETIC_REGISTRY):
        df = generate_synthetic_source(source_id, start, end, base_seed)
        if df.empty:
            continue
        save_dataframe_outputs(df, output_dir, source_id)
        frames.append(df)
        logger.info("  %s: %6d arrivals — %s", source_id, len(df),
                    SYNTHETIC_REGISTRY[source_id].description)

    if not frames:
        logger.error("No sources produced any data.")
        return False

    combined = pd.concat(frames, ignore_index=True)
    save_dataframe_outputs(combined, output_dir, "synthetic_arrivals")
    return True


def print_synthetic_registry() -> None:
    print(f"{'source_id':<10} description")
    print("-" * 80)
    for source_id in sorted(SYNTHETIC_REGISTRY):
        print(f"{source_id:<10} {SYNTHETIC_REGISTRY[source_id].description}")


# ═════════════════════════════════════════════════════════════════════════
# PART 2 — BTS airline on-time performance extraction
# ═════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True)
class RouteSpec:
    origin: str
    dest: str
    carrier: str

    @property
    def key(self) -> str:
        return f"{self.carrier}_{self.origin}-{self.dest}"

    @classmethod
    def parse(cls, value: str) -> RouteSpec:
        parts = value.strip().upper().split("-")
        if len(parts) != 3:
            raise argparse.ArgumentTypeError(
                f"Route must be formatted ORIGIN-DEST-CARRIER, got {value!r}"
            )
        origin, dest, carrier = parts
        return cls(origin=origin, dest=dest, carrier=carrier)


DEFAULT_ROUTES: List[RouteSpec] = [
    RouteSpec("JFK", "LAX", "DL"),
    RouteSpec("ORD", "ATL", "AA"),
    RouteSpec("BOS", "DCA", "B6"),
    RouteSpec("SEA", "SFO", "AS"),
    RouteSpec("MIA", "JFK", "AA"),
]


def hhmm_to_time(val, date: datetime) -> Optional[datetime]:
    try:
        v = int(val)
    except (TypeError, ValueError):
        return None
    if v == 2400:
        return datetime(date.year, date.month, date.day) + timedelta(days=1)
    hh, mm = divmod(v, 100)
    if hh > 23 or mm > 59:
        return None
    return datetime(date.year, date.month, date.day, hh, mm)


def _extract_bts_csv(zf: zipfile.ZipFile) -> Optional[pd.DataFrame]:
    csv_files = [n for n in zf.namelist() if n.lower().endswith(".csv") and "readme" not in n.lower()]
    if not csv_files:
        return None
    with zf.open(csv_files[0]) as f:
        return pd.read_csv(f, usecols=lambda c: c in BTS_KEEP_COLS, dtype=str, low_memory=False)


def download_bts_month(year: int, month: int) -> Optional[pd.DataFrame]:
    url = BTS_URL.format(year=year, month=month)
    logger.info("Downloading: %s", url)
    try:
        resp = requests.get(url, headers=BTS_HEADERS, timeout=120)
        resp.raise_for_status()
        if not resp.content.startswith(b"PK"):
            logger.warning("Skipped %d-%02d: response was not a ZIP file", year, month)
            return None
        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            df = _extract_bts_csv(zf)
        if df is None:
            logger.warning("No data CSV found for %d-%02d", year, month)
        return df
    except requests.RequestException as exc:
        logger.error("Download failed for %d-%02d: %s", year, month, exc)
        return None
    except zipfile.BadZipFile:
        logger.error("Corrupt ZIP downloaded for %d-%02d", year, month)
        return None


def load_bts_month(year: int, month: int, raw_dir: Path) -> Optional[pd.DataFrame]:
    zip_path = raw_dir / (
        f"On_Time_Reporting_Carrier_On_Time_Performance_1987_present_{year}_{month}.zip"
    )
    if not zip_path.exists():
        logger.warning("Missing local file: %s", zip_path)
        return None
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            df = _extract_bts_csv(zf)
        if df is None:
            logger.warning("No CSV found in %s", zip_path)
        return df
    except zipfile.BadZipFile:
        logger.error("Corrupt ZIP: %s", zip_path)
        return None


def load_or_download_month(year: int, month: int, raw_dir: Path, mode: str) -> Optional[pd.DataFrame]:
    if mode == "download":
        return download_bts_month(year, month)
    return load_bts_month(year, month, raw_dir)


def filter_route(df: pd.DataFrame, route: RouteSpec) -> pd.DataFrame:
    mask = (
            (df["Origin"] == route.origin)
            & (df["Dest"] == route.dest)
            & (df["Reporting_Airline"] == route.carrier)
    )
    return df[mask].copy()


def build_bts_source_id(row: pd.Series) -> str:
    return (
        f"{row['Reporting_Airline']}_"
        f"{str(row['Flight_Number_Reporting_Airline']).zfill(4)}_"
        f"{row['Origin']}-{row['Dest']}"
    )


def build_bts_pattern_description(row: pd.Series) -> str:
    """Mirrors the role of `pattern_description` on the synthetic side."""
    return (
        f"BTS Airline On-Time Performance — {row['Reporting_Airline']} flight "
        f"{str(row['Flight_Number_Reporting_Airline']).zfill(4)} "
        f"({row['Origin']} -> {row['Dest']})"
    )


def extract_bts_route(
        route: RouteSpec, years: Sequence[int], months: Sequence[int],
        raw_dir: Path, mode: str = "local",
) -> pd.DataFrame:
    """
    Returns a minimal, 3-column table — source_id, pattern_description,
    arrival_timestamp — matching the synthetic side's schema exactly.
    Cancelled flights are dropped here (a validity filter, not a derived
    feature: a cancelled flight has no real arrival event to record).
    """
    combos = [(y, m) for y in years for m in months]
    frames = []
    for year, month in tqdm(combos, desc=f"Loading months for {route.key}"):
        df_month = load_or_download_month(year, month, raw_dir, mode)
        if df_month is None:
            continue
        df_month = filter_route(df_month, route)
        if not df_month.empty:
            frames.append(df_month)

    if not frames:
        logger.error("No rows found for route %s in the requested period.", route.key)
        return pd.DataFrame()

    raw = pd.concat(frames, ignore_index=True)
    logger.info("Raw rows for %s after route filter: %d", route.key, len(raw))

    for col in ("Year", "Month", "DayofMonth", "Flight_Number_Reporting_Airline"):
        raw[col] = pd.to_numeric(raw[col], errors="coerce")
    raw["ArrTime"] = pd.to_numeric(raw["ArrTime"], errors="coerce")
    raw["Cancelled"] = pd.to_numeric(raw["Cancelled"], errors="coerce").fillna(0).astype(int)

    flight_date = pd.to_datetime(
        raw[["Year", "Month", "DayofMonth"]].rename(
            columns={"Year": "year", "Month": "month", "DayofMonth": "day"}),
        errors="coerce",
    )
    raw["actual_arrival_dt"] = [
        hhmm_to_time(arr_time, fdate) if pd.notna(fdate) else None
        for arr_time, fdate in zip(raw["ArrTime"], flight_date)
    ]

    clean = raw[(raw["Cancelled"] == 0) & raw["actual_arrival_dt"].notna()].copy()
    logger.info("Rows for %s after dropping cancellations: %d", route.key, len(clean))
    if clean.empty:
        return clean

    clean["source_id"] = clean.apply(build_bts_source_id, axis=1)
    clean["pattern_description"] = clean.apply(build_bts_pattern_description, axis=1)
    clean = clean.rename(columns={"actual_arrival_dt": "arrival_timestamp"})

    clean = (
        clean[["source_id", "pattern_description", "arrival_timestamp"]]
        .sort_values(["source_id", "arrival_timestamp"])
        .reset_index(drop=True)
    )
    return clean


def save_bts_outputs(df: pd.DataFrame, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = output_dir / f"{stem}.parquet"
    sample_path = output_dir / f"{stem}_sample.csv"
    df.to_parquet(parquet_path, index=False)
    df.head(10_000).to_csv(sample_path, index=False)
    logger.info("Saved %s (%d rows)", parquet_path, len(df))
    logger.info("Saved %s", sample_path)


def run_bts_single(route: RouteSpec, years: Sequence[int], months: Sequence[int],
                   raw_dir: Path, output_dir: Path, mode: str) -> bool:
    logger.info("Extracting BTS route %s for years=%s months=%s (mode=%s)",
                route.key, list(years), list(months), mode)
    clean = extract_bts_route(route, years, months, raw_dir, mode)
    if clean.empty:
        logger.error("No data extracted for route %s — nothing written.", route.key)
        return False
    save_bts_outputs(clean, output_dir, route.key)
    return True


def run_bts_all(routes: Sequence[RouteSpec], years: Sequence[int], months: Sequence[int],
                raw_dir: Path, output_dir: Path, mode: str) -> bool:
    results = [run_bts_single(route, years, months, raw_dir, output_dir, mode) for route in routes]
    return any(results)


def print_route_registry() -> None:
    print(f"{'route':<20} origin  dest  carrier")
    print("-" * 50)
    for route in DEFAULT_ROUTES:
        print(f"{route.key:<20} {route.origin:<7} {route.dest:<5} {route.carrier}")


# ═════════════════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════════════════
def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate file-arrival datasets, one source at a time or in bulk.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="Logging verbosity (default: INFO)")

    subparsers = parser.add_subparsers(dest="command", required=True)

    # ── synthetic ────────────────────────────────────────────────────────
    p_synth = subparsers.add_parser("synthetic", help="Generate synthetic ingestion sources")
    p_synth.add_argument("--source", type=str, default=None,
                         help="Source id (e.g. SRC_01) or 'all'. Required unless --list is given.")
    p_synth.add_argument("--start", type=parse_date, default=datetime(2023, 1, 1),
                         help="Start date, YYYY-MM-DD (default: 2023-01-01)")
    p_synth.add_argument("--end", type=parse_date, default=datetime(2024, 12, 31),
                         help="End date, YYYY-MM-DD (default: 2024-12-31)")
    p_synth.add_argument("--output-dir", type=Path, default=Path("data/synthetic"),
                         help="Directory to write output files (default: data/synthetic)")
    p_synth.add_argument("--seed", type=int, default=BASE_SEED,
                         help="Base seed for deterministic per-source randomness (default: 42)")
    p_synth.add_argument("--list", action="store_true",
                         help="List available source ids and exit")

    # ── bts ──────────────────────────────────────────────────────────────
    p_bts = subparsers.add_parser("bts", help="Extract BTS airline on-time performance data")
    p_bts.add_argument("--route", type=str, default=None,
                       help="Route as ORIGIN-DEST-CARRIER (e.g. JFK-LAX-DL) or 'all'. "
                            "Required unless --list is given.")
    p_bts.add_argument("--years", type=int, nargs="+", default=[2023, 2024],
                       help="Years to extract (default: 2023 2024)")
    p_bts.add_argument("--months", type=int, nargs="+", default=list(range(1, 13)),
                       help="Months to extract, 1-12 (default: all 12)")
    p_bts.add_argument("--raw-dir", type=Path, default=Path("data/airline_flights_raw"),
                       help="Directory containing locally staged monthly ZIPs "
                            "(used when --mode local)")
    p_bts.add_argument("--output-dir", type=Path, default=Path("data/bts"),
                       help="Directory to write output files (default: data/bts)")
    p_bts.add_argument("--mode", choices=["local", "download"], default="local",
                       help="Read monthly files from --raw-dir, or download them "
                            "from BTS TranStats (default: local)")
    p_bts.add_argument("--list", action="store_true",
                       help="List the predefined routes and exit")

    return parser


def cmd_synthetic(args: argparse.Namespace) -> int:
    if args.list:
        print_synthetic_registry()
        return 0
    if not args.source:
        logger.error("--source is required (a source id, or 'all'). Use --list to see options.")
        return 2
    if args.start > args.end:
        logger.error("--start (%s) must not be after --end (%s)", args.start.date(), args.end.date())
        return 2

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.source.lower() == "all":
        ok = run_synthetic_all(args.start, args.end, args.output_dir, args.seed)
    else:
        source_id = args.source.upper()
        if source_id not in SYNTHETIC_REGISTRY:
            logger.error("Unknown source id %r. Use --list to see valid ids.", args.source)
            return 2
        ok = run_synthetic_single(source_id, args.start, args.end, args.output_dir, args.seed)
    return 0 if ok else 1


def cmd_bts(args: argparse.Namespace) -> int:
    if args.list:
        print_route_registry()
        return 0
    if not args.route:
        logger.error("--route is required (ORIGIN-DEST-CARRIER, or 'all'). Use --list to see options.")
        return 2

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.route.lower() == "all":
        ok = run_bts_all(DEFAULT_ROUTES, args.years, args.months, args.raw_dir, args.output_dir, args.mode)
    else:
        try:
            route = RouteSpec.parse(args.route)
        except argparse.ArgumentTypeError as exc:
            logger.error(str(exc))
            return 2
        ok = run_bts_single(route, args.years, args.months, args.raw_dir, args.output_dir, args.mode)
    return 0 if ok else 1


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    configure_logging(args.log_level)

    try:
        if args.command == "synthetic":
            return cmd_synthetic(args)
        elif args.command == "bts":
            return cmd_bts(args)
        else:  # pragma: no cover - argparse enforces valid choices
            parser.error(f"Unknown command: {args.command}")
            return 2
    except Exception:
        logger.exception("Unhandled error while running %r", args.command)
        return 1


if __name__ == "__main__":
    sys.exit(main())
