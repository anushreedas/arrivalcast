"""
Single entry point: ``arrivalcast <command> [args]`` (or ``python -m arrivalcast``).

    arrivalcast data synthetic --source all       # generate the 26 synthetic sources
    arrivalcast data bts --route JFK-LAX-DL       # extract the BTS flight-arrival sources
    arrivalcast run --run-name baseline ...       # fit + evaluate all models
    arrivalcast tune-summary RUN [RUN ...] ...    # rank configs on validation loss
    arrivalcast compare REF RUN [RUN ...] ...     # paired ablation comparison
    arrivalcast report RUN_SYN [--bts RUN_BTS]    # tables + figures for the paper
    arrivalcast predict --checkpoint X --history Y  # next-arrival forecast
"""
from __future__ import annotations

import argparse
import json
import sys

COMMANDS = {
    "data": ("arrivalcast.data.pipeline", "Generate / extract datasets"),
    "run": ("arrivalcast.experiments.runner", "Fit and evaluate models"),
    "tune-summary": ("arrivalcast.experiments.tune_summary", "Rank configurations on validation loss"),
    "compare": ("arrivalcast.experiments.compare_runs", "Compare ablation runs with a reference run"),
    "report": ("arrivalcast.experiments.report", "Build paper tables and figures from run folders"),
    "hawkes-check": ("arrivalcast.experiments.hawkes_check", "Diagnose the Hawkes fit on the self-exciting source"),
    "predict": (None, "Forecast the next arrival from a history file"),
}


def _predict(argv: list[str]) -> int:
    import pandas as pd

    from arrivalcast.data.io import TS_COL
    from arrivalcast.inference import LogNormMixForecaster

    p = argparse.ArgumentParser(prog="arrivalcast predict", description="Next-arrival forecast (LogNormMix).")
    p.add_argument("--checkpoint", required=True, help="results/<run>/models/<source>__lnm_s<seed>.pt")
    p.add_argument("--history", required=True, help="CSV/Parquet with an arrival_timestamp column")
    p.add_argument("--source", default=None, help="filter the history to this source_id")
    p.add_argument("--sla-hours", type=float, default=None)
    p.add_argument("--as-of", default=None, help="current time (for the conditional SLA probability)")
    a = p.parse_args(argv)

    df = pd.read_parquet(a.history) if a.history.endswith(".parquet") else pd.read_csv(a.history)
    if a.source is not None:
        df = df[df["source_id"] == a.source]
    if df.empty:
        print("no history rows", file=sys.stderr)
        return 2
    ts = pd.to_datetime(df[TS_COL], utc=True, format="ISO8601").dt.tz_localize(None)
    fc = LogNormMixForecaster.from_checkpoint(a.checkpoint).forecast(
        ts, sla_hours=a.sla_hours, as_of=pd.Timestamp(a.as_of) if a.as_of else None)
    print(json.dumps(fc.to_dict(), indent=2, default=str))
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        print("commands:")
        for name, (_, desc) in COMMANDS.items():
            print(f"  {name:<14}{desc}")
        return 0
    cmd, rest = argv[0], argv[1:]
    if cmd not in COMMANDS:
        print(f"unknown command '{cmd}'. Choose from: {', '.join(COMMANDS)}", file=sys.stderr)
        return 2
    if cmd == "predict":
        return _predict(rest)
    import importlib

    module = importlib.import_module(COMMANDS[cmd][0])
    rc = module.main(rest)
    return int(rc or 0)


if __name__ == "__main__":
    sys.exit(main())
