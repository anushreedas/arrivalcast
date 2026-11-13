"""
Compare ablation runs against a reference run, per source.

    arrivalcast compare results/abl_full results/abl_nocal results/abl_not2v \
        --labels full no-calendar no-time2vec --model NeuralHawkes

The FIRST run is the reference. For every other run, for each source, the metric is
averaged over seeds and differenced against the reference:

    delta NLL  = NLL(ablation) - NLL(reference)      > 0  => the ablation is WORSE
    CRPS ratio = CRPS(ablation) / CRPS(reference)    > 1  => the ablation is WORSE

so a positive delta / ratio above 1 means the removed component was helping.
Sources flagged low_n (few test events) are skipped unless --include-low-n.
Compare each delta with the seed-to-seed spread printed alongside it: a difference
smaller than the spread is not evidence of anything.

Writes <out>/ablation_summary_<model>.csv, ablation_per_source_<model>.csv and
ablation_delta_nll_<model>.png (the model name is in the filename so runs for different models do not overwrite each other).
"""
from __future__ import annotations

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


def load(run_dir: str, model: str, include_low_n: bool) -> pd.DataFrame:
    m = pd.read_csv(os.path.join(run_dir, "full_metric_evaluation.csv"))
    m = m[m.model == model]
    if m.empty:
        raise SystemExit(f"{run_dir}: no rows for model '{model}' (models present: "
                         f"{sorted(pd.read_csv(os.path.join(run_dir, 'full_metric_evaluation.csv')).model.unique())})")
    if not include_low_n:
        m = m[~m.low_n]
    g = m.groupby("source_id").agg(nll=("mean_nll", "mean"), nll_sd=("mean_nll", "std"),
                                   crps=("mean_crps", "mean"), n_seeds=("mean_nll", "size"))
    return g


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="run directories; the first is the reference")
    ap.add_argument("--labels", nargs="*", default=None, help="one label per run (default: directory names)")
    ap.add_argument("--model", default="NeuralHawkes", choices=["NeuralHawkes", "LogNormMix", "Poisson", "Hawkes"])
    ap.add_argument("--include-low-n", action="store_true")
    ap.add_argument("--out", default=None, help="output folder (default: next to the reference run)")
    a = ap.parse_args(argv)

    labels = a.labels or [os.path.basename(os.path.normpath(r)) for r in a.runs]
    if len(labels) != len(a.runs):
        raise SystemExit("--labels must have one entry per run")
    out = a.out or os.path.join(a.runs[0], "ablations")
    os.makedirs(out, exist_ok=True)

    tabs = [load(r, a.model, a.include_low_n) for r in a.runs]
    ref, ref_label = tabs[0], labels[0]

    rows, per_source = [], {}
    for tab, lab in zip(tabs[1:], labels[1:]):
        common = ref.index.intersection(tab.index)
        if len(common) == 0:
            print(f"{lab}: no sources in common with the reference - skipped")
            continue
        d_nll = (tab.nll - ref.nll).loc[common]
        ratio = (tab.crps / ref.crps).loc[common]
        per_source[lab] = d_nll
        try:
            p = stats.wilcoxon(d_nll).pvalue
        except ValueError:
            p = np.nan
        rows.append({"ablation": lab, "sources": len(common),
                     "median dNLL (nats)": d_nll.median(), "worse on (NLL)": f"{int((d_nll > 0).sum())}/{len(common)}",
                     "median CRPS ratio": ratio.median(), "worse on (CRPS)": f"{int((ratio > 1).sum())}/{len(common)}",
                     "Wilcoxon p (NLL)": p,
                     "median seed SD of ref NLL": ref.nll_sd.loc[common].median(),
                     "seeds (ref/abl)": f"{int(ref.n_seeds.loc[common].median())}/{int(tab.n_seeds.loc[common].median())}"})
    if not rows:
        raise SystemExit("nothing to compare")

    summary = pd.DataFrame(rows).set_index("ablation")
    pd.set_option("display.width", 220); pd.set_option("display.max_columns", None)
    print(f"\n{a.model}: reference = '{ref_label}'   (delta > 0 or ratio > 1 means the ablation is WORSE)\n")
    print(summary.round(3).to_string())
    print("\nWilcoxon p-values are descriptive only (hand-built sources, few seeds).")
    summary.round(4).to_csv(os.path.join(out, f"ablation_summary_{a.model}.csv"))
    ps = pd.DataFrame(per_source); ps.index.name = "source_id"
    ps.round(4).to_csv(os.path.join(out, f"ablation_per_source_{a.model}.csv"))

    fig, ax = plt.subplots(figsize=(max(8, 0.55 * len(ps) + 3), 4.2))
    width = 0.8 / len(ps.columns)
    for i, c in enumerate(ps.columns):
        ax.bar(np.arange(len(ps)) + (i - (len(ps.columns) - 1) / 2) * width, ps[c], width=width, label=c, zorder=3)
    ax.axhline(0, color="black", lw=0.8)
    ax.axhspan(-ref.nll_sd.median(), ref.nll_sd.median(), color="grey", alpha=0.15, zorder=1,
               label="median seed SD of reference")
    ax.set_xticks(range(len(ps))); ax.set_xticklabels(ps.index, rotation=90)
    ax.set_ylabel(f"NLL(ablation) - NLL({ref_label})\n>0 = ablation worse")
    ax.set_title(f"{a.model}: effect of removing a component, per source", fontsize=10)
    ax.grid(axis="x", visible=False); ax.spines[["top", "right"]].set_visible(False); ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(out, f"ablation_delta_nll_{a.model}.png"), dpi=200)
    print(f"\nwrote ablation_summary_{a.model}.csv, ablation_per_source_{a.model}.csv, ablation_delta_nll_{a.model}.png in {out}/")


if __name__ == "__main__":
    main()
