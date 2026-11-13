"""
Run the full experiment: fit all four models per source, then compute the
canonical metrics (NLL, CRPS, PICP, SLA-hit, MAE vs. naive) on the test split.

Usage (from the project root):
    arrivalcast run                                  # defaults, seed 0
    arrivalcast run --seeds 0 1 2 --nhp-max-epochs 600
    arrivalcast run --sources SRC_01 SRC_08 --models lnm nhp
    arrivalcast run --run-name tune_lr1e-3 --lr 1e-3 --max-epochs 300
    arrivalcast run --lnm-lr 0.003 --nhp-max-epochs 600       # settings for ONE model: --lnm-* / --nhp-*
    arrivalcast run --dataset bts --seeds 0 1 2 --nhp-max-epochs 600   # BTS flights
    arrivalcast run --models nhp --no-calendar --run-name abl_nocal    # ablation
    arrivalcast run --models nhp --no-time2vec --run-name abl_not2v    # ablation
    arrivalcast run --dataset bts --freeze-from results/<tuned synthetic run> --run-name bts_final \
        --seeds 0 1 2 3 4 --models poisson hawkes nhp          # held-out BTS check with FROZEN settings

Every invocation writes to  results/<run-name>/ :
    config.json                    the TrainConfig + settings that produced the run
    test_predictions.parquet       one row per (source, model, seed, test event): actual gap,
                                   predictive quantiles, PIT, NLL, CRPS, arrival timestamps
    predictions/                   the same, one file per run (what the combined file is built from)
    models/                        one .pt checkpoint per neural run (load with
                                   training.load_lnm / training.load_nhp)
    parts/                         one JSON per finished (source, model, seed): re-running
                                   the same command resumes where it stopped
    model_comparison_results.csv   quick-pass diagnostics (not comparable across models)
    full_metric_evaluation.csv     canonical per-source / per-model / per-seed metrics
    training_log.csv               epochs, best epoch, steps, hit_cap, seconds per neural run

The default run name is derived from a hash of the training settings, so changing a
hyper-parameter automatically starts a fresh directory instead of silently reusing
old results. Re-using a run name with DIFFERENT settings is refused.
"""
from __future__ import annotations

import argparse
import dataclasses
import glob
import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
import torch

from arrivalcast import training as T
from arrivalcast.data.io import load_all_sources
from arrivalcast.evaluation.evaluate import source_metrics_to_row
from arrivalcast.models.hawkes.dataloader import HawkesDataLoader
from arrivalcast.models.hawkes.evaluate import evaluate_hawkes_source
from arrivalcast.models.hawkes.model import HawkesExponential
from arrivalcast.models.lognormmix.evaluate import evaluate_lognormmix_source
from arrivalcast.models.neural_hawkes.evaluate import evaluate_nhp_source
from arrivalcast.models.poisson.dataloader import PoissonDataLoader
from arrivalcast.models.poisson.evaluate import evaluate_poisson_source
from arrivalcast.models.poisson.model import HomogeneousPoisson

warnings.filterwarnings("ignore")
torch.set_num_threads(1)

# End of the observation period PER DATASET (used for the right-censored tail).
# BTS needs its own entry before it can be run: set it to the real extraction cutoff.
STUDY_END = {"synthetic": pd.Timestamp("2025-01-01")}
SLA_HOURS = 24.0
LOW_N_THRESHOLD = 5
ALL_MODELS = ("poisson", "hawkes", "nhp", "lnm")


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _jsonable(o):
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if hasattr(o, "item"):
        return o.item()
    return str(o)


def save_predictions(mm, source_id, model_name, seed, ds, pred_dir, tag):
    """Write one run's per-event test predictions (SourceMetrics.per_event) to Parquet."""
    if mm is None or mm.per_event is None:
        return
    df = mm.per_event.copy()
    df.insert(0, "dataset", ds)
    df.insert(0, "seed", pd.array([seed] * len(df), dtype="Int64"))  # <NA> for Poisson/Hawkes
    df.insert(0, "model", model_name)
    df.insert(0, "source_id", source_id)
    df.to_parquet(os.path.join(pred_dir, f"{source_id}__{tag}.parquet"), index=False)


def study_end_for(ds: str) -> pd.Timestamp:
    if ds not in STUDY_END:
        raise KeyError(f"No study end configured for dataset '{ds}'. Add it to STUDY_END "
                       f"in arrivalcast/experiments/runner.py (the real end of the observation period).")
    return STUDY_END[ds]


# ---------------------------------------------------------------- one unit of work
def _empty():
    return {"results": [], "metrics": [], "train_log": [], "failures": []}


def run_poisson(source_id, df, ds, pred_dir):
    out, end, ts = _empty(), study_end_for(ds), df["arrival_timestamp"]
    sp = PoissonDataLoader(ts, global_study_end=end).get_splits()
    if sp["train"].n_events >= 2 and sp["test"].n_events > 0:
        m = HomogeneousPoisson().fit(sp["train"].n_events, sp["train"].total_hours)
        ll = m.log_likelihood(sp["test"].n_events, sp["test"].total_hours)
        out["results"].append(dict(source_id=source_id, dataset=ds, model="Poisson",
                                   test_log_likelihood=ll, test_ll_per_event=ll / sp["test"].n_events,
                                   predicted_next_gap_hrs=m.predict_next_gap_hours()))
    try:
        mm = evaluate_poisson_source(source_id, ts, end, sla_hours=SLA_HOURS, low_n_threshold=LOW_N_THRESHOLD)
        if mm is not None:
            row = source_metrics_to_row(mm); row["dataset"] = ds; row["seed"] = None
            out["metrics"].append(row)
            save_predictions(mm, source_id, "Poisson", None, ds, pred_dir, "poisson")
    except Exception as e:
        out["failures"].append((source_id, "Poisson", repr(e)))
    return out


def run_hawkes(source_id, df, ds, pred_dir):
    out, end, ts = _empty(), study_end_for(ds), df["arrival_timestamp"]
    sh = HawkesDataLoader(ts, global_study_end=end).get_splits()
    if len(sh["train"].gaps_hours) >= 5 and len(sh["test"].gaps_hours) > 0:
        m = HawkesExponential().fit(sh["train"].gaps_hours, sh["train"].total_hours)
        ll = m.log_likelihood(sh["test"].gaps_hours, sh["test"].total_hours)
        p = m.params
        out["results"].append(dict(source_id=source_id, dataset=ds, model="Hawkes",
                                   test_log_likelihood=ll, test_ll_per_event=ll / len(sh["test"].gaps_hours),
                                   mu=p.mu, alpha=p.alpha, beta=p.beta, branching_ratio=p.branching_ratio))
    try:
        mm = evaluate_hawkes_source(source_id, ts, end, sla_hours=SLA_HOURS, low_n_threshold=LOW_N_THRESHOLD)
        if mm is not None:
            row = source_metrics_to_row(mm); row["dataset"] = ds; row["seed"] = None
            out["metrics"].append(row)
            save_predictions(mm, source_id, "Hawkes", None, ds, pred_dir, "hawkes")
    except Exception as e:
        out["failures"].append((source_id, "Hawkes", repr(e)))
    return out


def run_nhp(source_id, df, ds, cfg, seed, models_dir, pred_dir):
    out, end, ts = _empty(), study_end_for(ds), df["arrival_timestamp"]
    cfg = dataclasses.replace(cfg, seed=seed)
    t0 = time.time()
    res = T.train_nhp_one_source(df, end, cfg)
    if res is None:
        return out
    test_ll, n_test, model, t2v, info = res
    secs = round(time.time() - t0, 1)
    T.save_checkpoint(os.path.join(models_dir, f"{source_id}__nhp_s{seed}.pt"), "nhp", cfg,
                      info.pop("_input_dim"), model=model, time2vec=t2v)
    out["train_log"].append(dict(source_id=source_id, model="NeuralHawkes", seed=seed, **info, train_seconds=secs))
    out["results"].append(dict(source_id=source_id, dataset=ds, model="NeuralHawkes", seed=seed,
                               test_log_likelihood=test_ll, test_ll_per_event=test_ll / max(n_test, 1)))
    try:
        mm = evaluate_nhp_source(source_id, ts, end, model, t2v, use_calendar_features=cfg.use_calendar_features,
                                 sla_hours=SLA_HOURS, low_n_threshold=LOW_N_THRESHOLD)
        if mm is not None:
            row = source_metrics_to_row(mm); row["dataset"] = ds; row["seed"] = seed
            out["metrics"].append(row)
            save_predictions(mm, source_id, "NeuralHawkes", seed, ds, pred_dir, f"nhp_s{seed}")
    except Exception as e:
        out["failures"].append((source_id, "NeuralHawkes", repr(e)))
    log(f"{source_id} NHP s{seed}  epochs={info['epochs_run']:4d} best@{info['best_epoch']:4d} "
        f"steps={info['n_steps']:5d} {'CAP' if info['hit_cap'] else 'ok '}  {secs:6.1f}s")
    return out


def run_lnm(source_id, df, ds, cfg, seed, models_dir, pred_dir):
    out, end, ts = _empty(), study_end_for(ds), df["arrival_timestamp"]
    cfg = dataclasses.replace(cfg, seed=seed)
    t0 = time.time()
    res = T.train_lnm_one_source(df, end, cfg)
    if res is None:
        return out
    test_ll, n_pred, model, info = res
    secs = round(time.time() - t0, 1)
    T.save_checkpoint(os.path.join(models_dir, f"{source_id}__lnm_s{seed}.pt"), "lnm", cfg,
                      info.pop("_input_dim"), model=model)
    out["train_log"].append(dict(source_id=source_id, model="LogNormMix", seed=seed, **info, train_seconds=secs))
    out["results"].append(dict(source_id=source_id, dataset=ds, model="LogNormMix", seed=seed,
                               test_log_likelihood=test_ll, test_ll_per_event=test_ll / max(n_pred, 1)))
    try:
        mm = evaluate_lognormmix_source(source_id, ts, end, model, use_calendar_features=cfg.use_calendar_features,
                                        sla_hours=SLA_HOURS, low_n_threshold=LOW_N_THRESHOLD)
        if mm is not None:
            row = source_metrics_to_row(mm); row["dataset"] = ds; row["seed"] = seed
            out["metrics"].append(row)
            save_predictions(mm, source_id, "LogNormMix", seed, ds, pred_dir, f"lnm_s{seed}")
    except Exception as e:
        out["failures"].append((source_id, "LogNormMix", repr(e)))
    log(f"{source_id} LNM s{seed}  epochs={info['epochs_run']:4d} best@{info['best_epoch']:4d} "
        f"steps={info['n_steps']:5d} {'CAP' if info['hit_cap'] else 'ok '}  {secs:6.1f}s")
    return out


# ---------------------------------------------------------------- CLI / orchestration
def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-name", default=None, help="results/<run-name>/ (default: hash of the settings)")
    p.add_argument("--out-dir", default="results")
    p.add_argument("--sources", nargs="*", default=None, help="subset of source ids (default: all)")
    p.add_argument("--models", nargs="*", default=list(ALL_MODELS), choices=ALL_MODELS)
    p.add_argument("--seeds", nargs="*", type=int, default=[0], help="seeds for the neural models")
    p.add_argument("--dataset", choices=["synthetic", "bts", "all"], default="synthetic",
                   help="which data to run (default: synthetic)")
    p.add_argument("--synthetic-path", default="data/synthetic")
    p.add_argument("--bts-path", default="data/bts/DL_JFK-LAX.parquet")
    p.add_argument("--bts-study-end", default=None, metavar="YYYY-MM-DD",
                   help="end of the BTS observation period (extraction date). Only the censored tail of the "
                        "TEST split uses it, so it affects the quick-pass test_log_likelihood, not training or the "
                        "canonical metrics. Default: day after the latest BTS arrival (a warning is printed).")
    # training settings (None = keep TrainConfig default)
    p.add_argument("--hidden-dim", type=int); p.add_argument("--t2v-dim", type=int)
    p.add_argument("--n-components", type=int); p.add_argument("--n-mc-samples", type=int)
    p.add_argument("--lr", type=float); p.add_argument("--weight-decay", type=float)
    p.add_argument("--grad-clip", type=float)
    p.add_argument("--max-epochs", type=int); p.add_argument("--patience", type=int)
    p.add_argument("--chunk-len", type=int); p.add_argument("--burn-in", type=int)
    p.add_argument("--batch-size", type=int)
    p.add_argument("--nhp-max-epochs", type=int, default=None,
                   help="separate epoch cap for the Neural Hawkes (it converges more slowly)")
    p.add_argument("--allow-bts-tuning", action="store_true",
                   help="override the guard that keeps BTS a held-out check (BTS normally requires --freeze-from)")
    p.add_argument("--no-calendar", action="store_true", help="disable calendar features (ablation)")
    p.add_argument("--freeze-from", metavar="RUN_DIR", default=None,
                   help="use EXACTLY the LogNormMix/Neural-Hawkes settings recorded in RUN_DIR/config.json "
                        "(e.g. your final synthetic run) - for a held-out BTS evaluation. Any training flag "
                        "given alongside it is an error, so BTS cannot be run with re-tuned settings.")
    p.add_argument("--no-time2vec", action="store_true",
                   help="Neural Hawkes only: feed the raw scaled gap instead of Time2Vec (ablation)")
    # Per-model overrides (--lnm-lr, --nhp-lr, --lnm-hidden-dim, ...). The plain flag above applies to BOTH
    # models; a per-model flag overrides it for that model only. Needed when tuning picks different values.
    for name, typ in _TUNABLES.items():
        for model in ("lnm", "nhp"):
            flag = f"--{model}-{name.replace('_', '-')}"
            if flag == "--nhp-max-epochs":
                continue  # already defined above
            p.add_argument(flag, type=typ, default=None, help=f"{name.replace('_', ' ')} for {model.upper()} only")
    return p.parse_args(argv)


_TUNABLES = {"hidden_dim": int, "t2v_dim": int, "n_components": int, "n_mc_samples": int, "lr": float,
             "weight_decay": float, "grad_clip": float, "max_epochs": int, "patience": int, "chunk_len": int,
             "burn_in": int, "batch_size": int}
_TRAIN_FLAGS = tuple(_TUNABLES) + ("nhp_max_epochs",) + tuple(f"{m}_{k}" for k in _TUNABLES for m in ("lnm", "nhp")
                                                               if f"{m}_{k}" != "nhp_max_epochs")


def build_config(args) -> tuple[T.TrainConfig, T.TrainConfig]:
    if args.freeze_from:
        clash = [k for k in _TRAIN_FLAGS if getattr(args, k) is not None]
        clash += [k for k in ("no_calendar", "no_time2vec") if getattr(args, k)]
        if clash:
            sys.exit("--freeze-from fixes all training settings, so these flags cannot also be given: "
                     + ", ".join("--" + c.replace("_", "-") for c in clash))
        path = os.path.join(args.freeze_from, "config.json")
        if not os.path.exists(path):
            sys.exit(f"--freeze-from: {path} not found")
        rec = json.load(open(path))
        if "bts" in rec.get("study_end", {}):
            sys.exit(f"--freeze-from: {args.freeze_from} contains BTS results. Freeze settings from a synthetic-only "
                     f"run that was chosen without looking at BTS.")
        known = {f.name for f in dataclasses.fields(T.TrainConfig)}
        pick = lambda c: T.TrainConfig(**{k: v for k, v in c.items() if k in known})
        return pick(rec["lnm"]), pick(rec["nhp"])
    shared = {k: getattr(args, k) for k in _TUNABLES if getattr(args, k) is not None}
    if args.no_calendar:
        shared["use_calendar_features"] = False
    if args.no_time2vec:
        shared["use_time2vec"] = False
    base = dataclasses.replace(T.DEFAULT_CONFIG, **shared)
    per_model = {m: {k: getattr(args, f"{m}_{k}") for k in _TUNABLES if getattr(args, f"{m}_{k}", None) is not None}
                 for m in ("lnm", "nhp")}
    if args.nhp_max_epochs:
        per_model["nhp"]["max_epochs"] = args.nhp_max_epochs
    return dataclasses.replace(base, **per_model["lnm"]), dataclasses.replace(base, **per_model["nhp"])


def main(argv=None):
    args = parse_args(argv)
    if args.dataset in ("bts", "all") and not args.freeze_from and not args.allow_bts_tuning:
        sys.exit("BTS is reserved as a held-out check, so it must run with settings chosen on synthetic data:\n"
                 "    --freeze-from results/<tuned synthetic run>\n"
                 "(pass --allow-bts-tuning only if you deliberately want to experiment on BTS).")
    cfg_lnm, cfg_nhp = build_config(args)
    suffix = "" if args.dataset == "synthetic" else f"_{args.dataset}"
    run_name = args.run_name or f"cfg_{cfg_lnm.config_id()}_{cfg_nhp.config_id()}{suffix}"
    run_dir = os.path.join(args.out_dir, run_name)
    parts_dir = os.path.join(run_dir, "parts")
    models_dir = os.path.join(run_dir, "models")
    pred_dir = os.path.join(run_dir, "predictions")

    # ---- load data
    want_syn, want_bts = args.dataset in ("synthetic", "all"), args.dataset in ("bts", "all")
    if want_bts and not os.path.exists(args.bts_path):
        sys.exit(f"BTS file not found: {args.bts_path} (use --bts-path)")
    if want_syn and not os.path.exists(args.synthetic_path):
        sys.exit(f"Synthetic data not found: {args.synthetic_path} (use --synthetic-path)")
    sources = load_all_sources(synthetic_path=args.synthetic_path if want_syn else None,
                               bts_path=args.bts_path if want_bts else None)
    if not sources:
        sys.exit("No sources loaded - check --dataset / --synthetic-path / --bts-path.")
    if args.sources:
        missing = [s for s in args.sources if s not in sources]
        if missing:
            sys.exit(f"Unknown sources: {missing}")
        sources = {s: sources[s] for s in args.sources}

    # ---- study end per dataset (BTS has no hard-coded value)
    bts_dfs = [d for d in sources.values() if d.attrs.get("dataset") == "bts"]
    if bts_dfs:
        if args.bts_study_end:
            STUDY_END["bts"] = pd.Timestamp(args.bts_study_end)
        else:
            STUDY_END["bts"] = max(d["arrival_timestamp"].max() for d in bts_dfs).normalize() + pd.Timedelta(days=1)
            log(f"WARNING: --bts-study-end not given; using {STUDY_END['bts'].date()} (day after the latest BTS "
                f"arrival). Pass the real extraction date if you know it - it only changes the test-split tail term "
                f"in test_log_likelihood, not training or the canonical metrics.")
    used = {d.attrs["dataset"] for d in sources.values()}
    study_end = {k: str(v) for k, v in STUDY_END.items() if k in used}

    # ---- run directory + settings guard
    for d in (parts_dir, models_dir, pred_dir):
        os.makedirs(d, exist_ok=True)
    settings = {"lnm": cfg_lnm.to_dict(), "nhp": cfg_nhp.to_dict(), "study_end": study_end}
    if args.freeze_from:
        settings["frozen_from"] = args.freeze_from
        log(f"Settings frozen from {args.freeze_from} (no training flags applied).")
    if want_bts:
        others = []
        for cp in glob.glob(os.path.join(args.out_dir, "*", "config.json")):
            if os.path.abspath(os.path.dirname(cp)) == os.path.abspath(run_dir):
                continue
            try:
                prev_cfg = json.load(open(cp))
            except Exception:
                continue
            if "bts" in prev_cfg.get("study_end", {}) and any(prev_cfg.get(k) != settings[k] for k in ("lnm", "nhp")):
                others.append(os.path.basename(os.path.dirname(cp)))
        if others:
            log(f"WARNING: BTS has already been evaluated under DIFFERENT settings in: {', '.join(others)}. "
                f"Choosing between configurations by their BTS results is tuning on BTS; report them all, "
                f"or keep to one frozen configuration.")
    cfg_path = os.path.join(run_dir, "config.json")
    if os.path.exists(cfg_path):
        prev = json.load(open(cfg_path))
        if any(prev.get(k) != settings[k] for k in ("lnm", "nhp")):
            sys.exit(f"Run '{run_name}' already exists with different training settings ({cfg_path}). "
                     f"Pick another --run-name, or delete that directory to start over.")
        clash = {k: (prev["study_end"][k], v) for k, v in study_end.items() if k in prev.get("study_end", {}) and prev["study_end"][k] != v}
        if clash:
            sys.exit(f"Run '{run_name}' was made with a different study end {clash}. Pick another --run-name.")
        settings["study_end"] = {**prev.get("study_end", {}), **study_end}   # allow adding a dataset to a run
    json.dump(settings, open(cfg_path, "w"), indent=2, default=_jsonable)

    log(f"Run '{run_name}': {len(sources)} sources, models={args.models}, seeds={args.seeds}")
    log(f"LNM  config: {cfg_lnm.to_dict()}")
    if cfg_nhp != cfg_lnm:
        log(f"NHP  differs: max_epochs={cfg_nhp.max_epochs}")

    crashed = []   # units that raised; NOT cached, so re-running the command retries them

    def unit(name, fn, source_id):
        path = os.path.join(parts_dir, f"{source_id}__{name}.json")
        if os.path.exists(path):
            return
        try:
            res = fn()
        except Exception as e:  # one degenerate source must not kill a 56-source run
            log(f"ERROR {source_id} {name}: {type(e).__name__}: {e}")
            crashed.append((source_id, name, repr(e)))
            return
        with open(path, "w") as fh:
            json.dump(res, fh, default=_jsonable)

    for source_id, df in sources.items():
        ds = df.attrs["dataset"]
        if "poisson" in args.models:
            unit("poisson", lambda: run_poisson(source_id, df, ds, pred_dir), source_id)
        if "hawkes" in args.models:
            unit("hawkes", lambda: run_hawkes(source_id, df, ds, pred_dir), source_id)
        for seed in args.seeds:
            if "nhp" in args.models:
                unit(f"nhp_s{seed}", lambda: run_nhp(source_id, df, ds, cfg_nhp, seed, models_dir, pred_dir), source_id)
            if "lnm" in args.models:
                unit(f"lnm_s{seed}", lambda: run_lnm(source_id, df, ds, cfg_lnm, seed, models_dir, pred_dir), source_id)

    # ---- aggregate every finished part in this run directory
    # Summaries always cover EVERYTHING finished in this run folder, not just the sources named
    # in this invocation, so adding seeds/sources later never drops earlier results.
    agg = _empty()
    for path in sorted(glob.glob(os.path.join(parts_dir, "*.json"))):
        part = json.load(open(path))
        for k in agg:
            agg[k] += part[k]
    pd.DataFrame(agg["results"]).to_csv(os.path.join(run_dir, "model_comparison_results.csv"), index=False)
    metrics = pd.DataFrame(agg["metrics"])
    metrics.to_csv(os.path.join(run_dir, "full_metric_evaluation.csv"), index=False)
    tlog = pd.DataFrame(agg["train_log"])
    tlog.to_csv(os.path.join(run_dir, "training_log.csv"), index=False)

    pred_files = sorted(glob.glob(os.path.join(pred_dir, "*.parquet")))
    if pred_files:
        preds = pd.concat([pd.read_parquet(f) for f in pred_files], ignore_index=True)
        preds.to_parquet(os.path.join(run_dir, "test_predictions.parquet"), index=False)
        log(f"test_predictions.parquet: {len(preds):,} rows from {len(pred_files)} runs")

    log(f"{len(metrics)} (source, model, seed) rows evaluated; {len(agg['failures'])} failures -> {run_dir}/")
    for f in agg["failures"]:
        print("  FAILURE:", f)
    if crashed:
        log(f"{len(crashed)} unit(s) CRASHED and were skipped (re-running the same command retries them):")
        for c in crashed:
            print("  CRASH:", c)
    if len(tlog):
        for model, g in tlog.groupby("model"):
            log(f"{model}: {int(g.hit_cap.sum())}/{len(g)} runs ended at the epoch cap "
                f"(under-trained if > 0); median {g.train_seconds.median():.0f}s/run")
    if len(metrics):
        per_src = metrics.groupby(["model", "source_id"]).mean_nll.mean().reset_index()
        log("median over sources of (seed-mean) test NLL:\n" + per_src.groupby("model").mean_nll.median().round(3).to_string())


if __name__ == "__main__":
    sys.exit(main())
