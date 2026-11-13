"""
Choose hyper-parameters on VALIDATION loss only (never on test metrics).

    arrivalcast tune-summary results/tune_ref results/tune_lr3e-3 results/tune_h64 \
        --labels ref lr3e-3 h64 --model lnm

Each run folder is one configuration, run on the same synthetic sources. The FIRST run is the
reference. For every source the validation loss at the best epoch is averaged over seeds and
differenced against the reference (negative = the configuration fits validation data better):

    delta_i = val_i(config) - val_i(reference)

then summarised over sources. A configuration is only called "clearly better" when its mean
delta is more than 2 standard errors below zero; otherwise treat it as a tie and prefer the
cheaper / simpler one.

Cautions (printed again when they apply):
  * Validation sets are small. Use --min-val to keep only sources with enough validation events
    (needs the n_val column, present in runs made with the current training.py).
  * Validation losses are only comparable between runs with the same chunk_len and burn_in
    (they change how validation is computed), so those must not differ.
  * Values are in each model's own units; compare configs within one model only.
Writes tune_summary_<model>.csv next to where you run it (or --out).
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

MODELS = {"lnm": ("LogNormMix", "best_val_nll_per_gap"), "nhp": ("NeuralHawkes", "best_val_nll_per_event")}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="run folders; the first is the reference configuration")
    ap.add_argument("--labels", nargs="*", default=None)
    ap.add_argument("--model", choices=list(MODELS), required=True)
    ap.add_argument("--min-val", type=int, default=10, help="keep sources with at least this many validation events")
    ap.add_argument("--out", default=".")
    a = ap.parse_args(argv)

    model_name, val_col = MODELS[a.model]
    labels = a.labels or [os.path.basename(os.path.normpath(r)) for r in a.runs]
    if len(labels) != len(a.runs):
        raise SystemExit("--labels needs one entry per run")

    logs, cfgs = [], []
    for r in a.runs:
        t = pd.read_csv(os.path.join(r, "training_log.csv"))
        t = t[t.model == model_name]
        if t.empty:
            raise SystemExit(f"{r}: no {model_name} runs in training_log.csv")
        logs.append(t)
        cj = json.load(open(os.path.join(r, "config.json")))
        cfgs.append(cj[a.model])

    # ---- what differs between configurations, and is validation comparable?
    ref_cfg = cfgs[0]
    print(f"\n{model_name}: reference = '{labels[0]}'")
    for lab, c in zip(labels[1:], cfgs[1:]):
        diff = {k: (ref_cfg[k], c[k]) for k in c if k != "seed" and ref_cfg.get(k) != c[k]}
        print(f"  {lab}: differs from reference in {diff if diff else 'nothing (check your flags!)'}")
        if any(k in diff for k in ("chunk_len", "burn_in")):
            print(f"  WARNING: {lab} changes chunk_len/burn_in, so its validation loss is NOT comparable with the reference.")

    per_source = []
    for t in logs:
        g = t.groupby("source_id").agg(val=(val_col, "mean"), seeds=("seed", "nunique"), hit_cap=("hit_cap", "mean"),
                                       secs=("train_seconds", "median"),
                                       **({"n_val": ("n_val", "first")} if "n_val" in t else {}))
        per_source.append(g)

    common = set.intersection(*[set(g.index) for g in per_source])
    if all("n_val" in g for g in per_source):
        common = {s for s in common if per_source[0].loc[s, "n_val"] >= a.min_val}
        note = f"sources with >= {a.min_val} validation events"
    else:
        note = "all common sources (n_val missing from these logs, so --min-val could not be applied)"
    common = sorted(common)
    if len(common) < 3:
        raise SystemExit(f"only {len(common)} usable sources ({note}); lower --min-val or run more sources")
    print(f"  using {len(common)} {note}\n")

    ref = per_source[0].loc[common, "val"]
    rows = []
    for lab, g in zip(labels, per_source):
        d = (g.loc[common, "val"] - ref)
        se = d.std(ddof=1) / np.sqrt(len(d)) if lab != labels[0] else 0.0
        rows.append({"config": lab, "mean delta val": d.mean(), "median delta val": d.median(), "SE of mean": se,
                     "better on": f"{int((d < 0).sum())}/{len(d)}" if lab != labels[0] else "-",
                     "clearly better than ref": bool(d.mean() + 2 * se < 0) if lab != labels[0] else False,
                     "seeds": int(g.loc[common, "seeds"].median()),
                     "share of runs at epoch cap": g.loc[common, "hit_cap"].mean(),
                     "median train s": g.loc[common, "secs"].median()})
    out = pd.DataFrame(rows).set_index("config")
    pd.set_option("display.width", 220); pd.set_option("display.max_columns", None)
    print(out.round(3).to_string())
    best = out["mean delta val"].idxmin()
    print(f"\nLowest mean validation delta: '{best}'" +
          ("" if out.loc[best, "clearly better than ref"] or best == labels[0]
           else " - but NOT clearly better than the reference (within 2 SE): treat as a tie."))
    if (out["share of runs at epoch cap"] > 0.2).any():
        print("NOTE: some configs hit the epoch cap on >20% of runs; a better config may just be one that trains longer.")
    os.makedirs(a.out, exist_ok=True)
    out.round(5).to_csv(os.path.join(a.out, f"tune_summary_{a.model}.csv"))
    print(f"wrote {os.path.join(a.out, f'tune_summary_{a.model}.csv')}")


if __name__ == "__main__":
    main()
