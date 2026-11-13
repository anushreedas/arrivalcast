"""
Build every table, figure and headline number used in the paper from run folders.

    arrivalcast report results/final_syn --bts results/bts_final \
        --tuning results/tuning --data data/synthetic --out paper/generated

Nothing here trains or re-scores a model: it only reads what the runner wrote
(``full_metric_evaluation.csv``, ``training_log.csv``, ``model_comparison_results.csv``,
``test_predictions.parquet``), so the paper can be regenerated in seconds and every
number in it traces back to a CSV row.

Outputs (in --out):
    numbers.tex          \\newcommand macros for every number quoted in the text
    tab_*.tex            booktabs tables (\\input from the paper)
    fig_*.pdf            vector figures

Conventions (same as notebooks/results_analysis.ipynb):
    * neural models are averaged over seeds per source first;
    * sources flagged ``low_n`` (< 5 test events) are excluded from headline statistics;
    * raw NLL/CRPS scale with a source's cadence, so cross-source summaries use the
      NLL gain over Poisson (nats/event) and the CRPS skill 1 - CRPS/CRPS_Poisson.
"""
from __future__ import annotations

import argparse
import glob
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

MODELS = ["Poisson", "Hawkes", "NeuralHawkes", "LogNormMix"]
SHORT = {"Poisson": "Poisson", "Hawkes": "Hawkes", "NeuralHawkes": "NHP", "LogNormMix": "LogNormMix"}
# Okabe-Ito (colour-blind safe); markers give a second, non-colour cue.
COLORS = {"Poisson": "#8c8c8c", "Hawkes": "#e69f00", "NeuralHawkes": "#0072b2", "LogNormMix": "#009e73"}
MARKERS = {"Poisson": "o", "Hawkes": "s", "NeuralHawkes": "^", "LogNormMix": "D"}

SOURCE_GROUPS = {
    "Regular cadence": ["SRC_01", "SRC_08", "SRC_09"],
    "Calendar-driven": ["SRC_02", "SRC_03", "SRC_06", "SRC_07", "SRC_10", "SRC_11", "SRC_14"],
    "Low frequency": ["SRC_04", "SRC_05", "SRC_12", "SRC_13", "SRC_15", "SRC_16", "SRC_17", "SRC_18", "SRC_23"],
    "Perturbed": ["SRC_19", "SRC_20", "SRC_26"],
    "Bursty": ["SRC_21", "SRC_22", "SRC_24"],
    "Heavy-tailed": ["SRC_25"],
}
GROUP_OF = {s: g for g, ss in SOURCE_GROUPS.items() for s in ss}

plt.rcParams.update({
    "figure.dpi": 150, "savefig.bbox": "tight", "font.size": 8, "axes.titlesize": 8.5,
    "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25,
    "grid.linewidth": 0.5, "axes.axisbelow": True, "lines.linewidth": 1.5, "pdf.fonttype": 42,
})


# ----------------------------------------------------------------------------- loading
class Run:
    """One run folder, aggregated to one row per (source, model)."""

    def __init__(self, path: str):
        self.path = path
        self.metrics = pd.read_csv(os.path.join(path, "full_metric_evaluation.csv"))
        tl = os.path.join(path, "training_log.csv")
        self.tlog = pd.read_csv(tl) if os.path.exists(tl) and os.path.getsize(tl) > 1 else None
        mc = os.path.join(path, "model_comparison_results.csv")
        self.comparison = pd.read_csv(mc) if os.path.exists(mc) else None
        pp = os.path.join(path, "test_predictions.parquet")
        self.preds = pd.read_parquet(pp) if os.path.exists(pp) else None

        m = self.metrics
        self.agg = (m.groupby(["source_id", "model"], as_index=False).agg(
            n_test=("n_test_events", "first"), low_n=("low_n", "first"),
            nll=("mean_nll", "mean"), nll_sd=("mean_nll", "std"), crps=("mean_crps", "mean"),
            mae=("mae", "mean"), baseline_mae=("baseline_mae", "first"),
            cov50=("coverage_observed_50", "mean"), cov80=("coverage_observed_80", "mean"),
            cov90=("coverage_observed_90", "mean"), sla_acc=("sla_accuracy", "mean"),
            sla_f1=("sla_f1", "mean"), sla_prec=("sla_precision", "mean"), sla_rec=("sla_recall", "mean"),
            n_seeds=("model", "size")))
        self.agg["mae_ratio"] = self.agg.mae / self.agg.baseline_mae
        src = self.agg.drop_duplicates("source_id").set_index("source_id")[["n_test", "low_n"]]
        # a source is usable only if every model produced a row for it
        complete = self.agg.groupby("source_id").model.nunique() == len(MODELS)
        self.sources = sorted(src.index)
        self.valid = [s for s in self.sources if not src.loc[s, "low_n"] and complete.get(s, False)]
        self.low_n = sorted(src.index[src.low_n])

    def wide(self, col: str, valid_only: bool = True) -> pd.DataFrame:
        w = self.agg.pivot(index="source_id", columns="model", values=col)[MODELS]
        return w.loc[self.valid] if valid_only else w

    def seeds(self, model: str) -> int:
        return int(self.metrics[self.metrics.model == model].seed.nunique())


# ----------------------------------------------------------------------------- statistics
def scoreboard(run: Run) -> pd.DataFrame:
    nll, crps = run.wide("nll"), run.wide("crps")
    gain = -(nll.sub(nll["Poisson"], axis=0))
    skill = 1 - crps.div(crps["Poisson"], axis=0)
    rank = nll.rank(axis=1, method="min").mean()
    best_nll = nll.idxmin(axis=1).value_counts()
    best_crps = crps.idxmin(axis=1).value_counts()
    mae_ratio = run.wide("mae_ratio")
    informative = [s for s in run.valid if run.wide("sla_acc").loc[s, "Poisson"] < 0.95]
    rows = {}
    for m in MODELS:
        rows[m] = {
            "nll_gain": gain[m].median(), "crps_skill": skill[m].median(), "mean_rank": rank[m],
            "best_nll": int(best_nll.get(m, 0)), "best_crps": int(best_crps.get(m, 0)),
            "cov50": run.wide("cov50")[m].median(), "cov80": run.wide("cov80")[m].median(),
            "cov90": run.wide("cov90")[m].median(),
            "mae_ratio": mae_ratio[m].median(), "beats_lag1": int((mae_ratio[m] < 1).sum()),
            "sla_acc": run.wide("sla_acc").loc[informative, m].mean() if informative else np.nan,
        }
    out = pd.DataFrame(rows).T
    out.attrs["n_valid"], out.attrs["n_informative"] = len(run.valid), len(informative)
    return out


def paired_tests(run: Run) -> pd.DataFrame:
    nll, crps = run.wide("nll"), run.wide("crps")
    pairs = [("LogNormMix", "NeuralHawkes"), ("LogNormMix", "Poisson"), ("NeuralHawkes", "Poisson"),
             ("Hawkes", "Poisson")]
    rows = []
    for name, t in (("NLL", nll), ("CRPS", crps)):
        for a, b in pairs:
            d = (t[a] - t[b]).dropna()
            try:
                p = stats.wilcoxon(d).pvalue
            except ValueError:
                p = np.nan
            eff = -d.median() if name == "NLL" else (t[a] / t[b]).median()
            rows.append(dict(metric=name, a=a, b=b, wins=int((d < 0).sum()), n=len(d), effect=eff, p=p))
    return pd.DataFrame(rows)


def by_group(run: Run) -> pd.DataFrame:
    nll = run.wide("nll")
    gain = -(nll.sub(nll["Poisson"], axis=0))
    gain["group"] = [GROUP_OF.get(s, "Other") for s in gain.index]
    g = gain.groupby("group").agg(NHP=("NeuralHawkes", "median"), LNM=("LogNormMix", "median"),
                                  n=("LogNormMix", "size"))
    return g.reindex([k for k in SOURCE_GROUPS if k in g.index])


def point_stats(run: Run) -> dict:
    """Point-forecast and SLA statistics in the units an ingestion team reads (hours, %)."""
    mae, nll = run.wide("mae"), run.wide("nll")
    lag1 = run.agg[run.agg.model == "Poisson"].set_index("source_id").loc[run.valid].baseline_mae
    d = (mae["LogNormMix"] - mae["NeuralHawkes"]).dropna()
    try:
        p_mae = stats.wilcoxon(d).pvalue
    except ValueError:
        p_mae = np.nan
    big_loss = int(((nll["NeuralHawkes"] - nll["Poisson"]) > 1.0).sum())
    out = {"mae_med": mae.median(), "nll_mean": nll.mean(), "nll_med": nll.median(),
           "lag1_mae": float(lag1.median()), "lnm_mae_wins": int((d < 0).sum()), "mae_p": p_mae,
           "nhp_big_loss": big_loss,
           "n_test_med": float(run.agg[run.agg.model == "Poisson"].set_index("source_id").loc[run.valid].n_test.median()),
           "sla_acc_med": run.wide("sla_acc").median(), "sla_prec_med": run.wide("sla_prec").median(),
           "sla_rec_med": run.wide("sla_rec").median()}
    return out


def tex_group_mae(run: Run) -> str:
    mae = run.wide("mae")
    lag1 = run.agg[run.agg.model == "Poisson"].set_index("source_id").baseline_mae
    mae["Lag-1"] = lag1.reindex(mae.index)
    mae["group"] = [GROUP_OF.get(s, "Other") for s in mae.index]
    cols = ["Poisson", "NeuralHawkes", "LogNormMix", "Lag-1"]
    g = mae.groupby("group")[cols].median().reindex([k for k in SOURCE_GROUPS if k in set(mae.group)])
    n = mae.groupby("group").size()
    lines = [r"\begin{table}[t]", r"\centering\small",
             r"\caption{Median MAE (hours) of the median forecast by schedule type, synthetic data. ``Lag-1'' repeats "
             r"the previous gap. Hawkes is omitted (within 0.3\,h of Poisson in every group). Lowest value per row in bold.}",
             r"\label{tab:group-mae}", r"\resizebox{\columnwidth}{!}{%", r"\begin{tabular}{lccccc}", r"\toprule",
             r"Schedule type & $n$ & Poisson & NHP & LogNormMix & Lag-1 \\", r"\midrule"]
    for k, r in g.iterrows():
        best = r.idxmin()
        cells = [rf"\textbf{{{r[c]:.2f}}}" if c == best else f"{r[c]:.2f}" for c in cols]
        lines.append(f"{k} & {int(n[k])} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table}", ""]
    return "\n".join(lines)


def training_summary(run: Run) -> pd.DataFrame:
    if run.tlog is None:
        return pd.DataFrame()
    return run.tlog.groupby("model").agg(
        runs=("source_id", "size"), sources=("source_id", "nunique"), seeds=("seed", "nunique"),
        hit_cap=("hit_cap", "sum"), med_best=("best_epoch", "median"),
        med_sec=("train_seconds", "median"), total_min=("train_seconds", lambda s: s.sum() / 60))


# ----------------------------------------------------------------------------- LaTeX helpers
def fmt(x, d=2, sign=False):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "--"
    if round(x, d) == 0:
        x = 0.0  # avoid "-0.00"
    return f"{x:+.{d}f}" if sign else f"{x:.{d}f}"


def fmt_p(p):
    if not np.isfinite(p):
        return "--"
    return r"$<$0.001" if p < 1e-3 else f"{p:.3f}"


def write(path, text):
    with open(path, "w") as fh:
        fh.write(text)


def tex_scoreboard(sb: pd.DataFrame, caption: str, label: str) -> str:
    best = {c: (sb[c].idxmax() if c in ("nll_gain", "crps_skill") else sb[c].idxmin())
            for c in ("nll_gain", "crps_skill", "mean_rank", "mae_ratio")}

    def cell(m, c, d=2, sign=False):
        s = fmt(sb.loc[m, c], d, sign)
        return rf"\textbf{{{s}}}" if best.get(c) == m else s

    n = sb.attrs["n_valid"]
    lines = [r"\begin{table*}[t]", r"\centering\small", rf"\caption{{{caption}}}", rf"\label{{{label}}}",
             r"\begin{tabular}{lcccccccc}", r"\toprule",
             r" & \multicolumn{2}{c}{Probabilistic skill} & & \multicolumn{2}{c}{Sources won} & "
             r"\multicolumn{2}{c}{Interval coverage} & Point \\",
             r"\cmidrule(lr){2-3}\cmidrule(lr){5-6}\cmidrule(lr){7-8}\cmidrule(lr){9-9}",
             r"Model & NLL gain & CRPS skill & Mean rank & NLL & CRPS & 50\% & 80\% & MAE / lag-1 \\",
             r"\midrule"]
    for m in MODELS:
        lines.append(f"{SHORT[m]} & {cell(m, 'nll_gain', 2, True)} & {cell(m, 'crps_skill')} & "
                     f"{cell(m, 'mean_rank')} & {int(sb.loc[m, 'best_nll'])}/{n} & {int(sb.loc[m, 'best_crps'])}/{n} & "
                     f"{fmt(sb.loc[m, 'cov50'])} & {fmt(sb.loc[m, 'cov80'])} & {cell(m, 'mae_ratio')} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""]
    return "\n".join(lines)


def tex_paired(pt_syn: pd.DataFrame, pt_bts: pd.DataFrame | None) -> str:
    cols = "llcccccc" if pt_bts is not None else "llccc"
    head = (r"Metric & Comparison (A vs B) & \multicolumn{3}{c}{Synthetic} & \multicolumn{3}{c}{BTS flights} \\"
            if pt_bts is not None else r"Metric & Comparison (A vs B) & \multicolumn{3}{c}{Synthetic} \\")
    sub = r" & & A wins & Effect & $p$" + (r" & A wins & Effect & $p$" if pt_bts is not None else "") + r" \\"
    lines = [r"\begin{table*}[t]", r"\centering\small",
             r"\caption{Paired comparisons across sources (Wilcoxon signed-rank). ``Effect'' is the median NLL "
             r"advantage of A in nats/event for NLL and the median CRPS ratio A/B for CRPS ($<1$ favours A). "
             r"$p$-values are descriptive: the synthetic sources are designed scenarios, not a random sample.}",
             r"\label{tab:paired}", rf"\begin{{tabular}}{{{cols}}}", r"\toprule", head,
             r"\cmidrule(lr){3-5}" + (r"\cmidrule(lr){6-8}" if pt_bts is not None else ""), sub, r"\midrule"]
    for i, r in pt_syn.iterrows():
        eff_d = 2 if r.metric == "NLL" else 3
        row = (f"{r.metric} & {SHORT[r.a]} vs {SHORT[r.b]} & {r.wins}/{r.n} & "
               f"{fmt(r.effect, eff_d, r.metric == 'NLL')} & {fmt_p(r.p)}")
        if pt_bts is not None:
            b = pt_bts.iloc[i]
            row += f" & {b.wins}/{b.n} & {fmt(b.effect, eff_d, b.metric == 'NLL')} & {fmt_p(b.p)}"
        lines.append(row + r" \\")
        if i == 3:
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""]
    return "\n".join(lines)


def tex_groups(g: pd.DataFrame) -> str:
    lines = [r"\begin{table}[t]", r"\centering\small",
             r"\caption{Median NLL gain over Poisson (nats/event) by schedule type, synthetic data.}",
             r"\label{tab:groups}", r"\begin{tabular}{lccc}", r"\toprule",
             r"Schedule type & Sources & NHP & LogNormMix \\", r"\midrule"]
    for k, r in g.iterrows():
        lines.append(f"{k} & {int(r.n)} & {fmt(r.NHP, 2, True)} & {fmt(r.LNM, 2, True)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


def tex_training(ts_syn: pd.DataFrame, ts_bts: pd.DataFrame | None) -> str:
    lines = [r"\begin{table}[t]", r"\centering\small",
             r"\caption{Training cost and convergence of the neural models (CPU, one thread). ``At cap'' counts "
             r"runs that were still improving when the epoch budget ran out.}",
             r"\label{tab:training}", r"\resizebox{\columnwidth}{!}{%", r"\begin{tabular}{llcccc}", r"\toprule",
             r"Data & Model & Runs & At cap & Best epoch & Sec./run \\", r"\midrule"]
    for name, t in (("Synthetic", ts_syn), ("BTS", ts_bts)):
        if t is None or t.empty:
            continue
        for m in ("NeuralHawkes", "LogNormMix"):
            if m in t.index:
                r = t.loc[m]
                lines.append(f"{name} & {SHORT[m]} & {int(r.runs)} & {int(r.hit_cap)} & {int(r.med_best)} & "
                             f"{r.med_sec:.1f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table}", ""]
    return "\n".join(lines)


def tex_tuning(tuning_dir: str) -> str | None:
    rows = []
    for model, label in (("lnm", "LogNormMix"), ("nhp", "NHP")):
        p = os.path.join(tuning_dir, f"tune_summary_{model}.csv")
        if not os.path.exists(p):
            continue
        t = pd.read_csv(p)
        for _, r in t.iterrows():
            rows.append((label, r.config, r["mean delta val"], r["SE of mean"], r["better on"],
                         r["share of runs at epoch cap"], r["median train s"], int(r.seeds)))
    if not rows:
        return None
    desc = {"ref": "reference ($\\eta=10^{-2}$, $H=32$, $K=4$)", "lr3e-3": "$\\eta=3\\cdot10^{-3}$",
            "h64": "hidden size $H=64$", "k8": "$K=8$ mixture components"}
    lines = [r"\begin{table}[t]", r"\centering\small",
             r"\caption{Hyper-parameter selection on the \emph{validation} split only (synthetic sources with "
             r"$\geq$10 validation events). $\Delta$ is the mean change in validation NLL per event relative to the "
             r"reference ($<0$ is better), with its standard error over sources.}",
             r"\label{tab:tuning}", r"\resizebox{\columnwidth}{!}{%", r"\begin{tabular}{llccccc}", r"\toprule",
             r"Model & Configuration & $\Delta$ val.\ NLL & SE & Better on & At cap & Seeds \\", r"\midrule"]
    last = None
    for label, cfg, d, se, better, cap, _secs, seeds in rows:
        if last is not None and label != last:
            lines.append(r"\midrule")
        last = label
        d_txt = "reference ($\\eta=10^{-2}$, $H=32$)" if (label == "NHP" and cfg == "ref") else desc.get(cfg, cfg)
        lines.append(f"{label} & {d_txt} & {fmt(d, 3, True) if cfg != 'ref' else '0'} & "
                     f"{fmt(se, 3) if cfg != 'ref' else '--'} & {better if cfg != 'ref' else '--'} & {cap:.2f} & {seeds} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table}", ""]
    return "\n".join(lines)


def tex_sources(data_dir: str, run: Run) -> str | None:
    files = sorted(glob.glob(os.path.join(data_dir, "SRC_*.csv")))
    if not files:
        return None
    lines = [r"\begin{table*}[t]", r"\centering\scriptsize",
             r"\caption{The 26 synthetic sources (2023--2024). Median gap in hours; $n$ = arrivals; "
             r"test = completed test gaps scored. $\dagger$: fewer than 5 test events, excluded from headline "
             r"statistics; SRC\_23 (annual, 2 arrivals) cannot be modelled.}",
             r"\label{tab:sources}", r"\begin{tabular}{llp{9.2cm}rrr}", r"\toprule",
             r"Source & Group & Generating pattern & $n$ & Median gap & Test \\", r"\midrule"]
    ntest = run.agg.groupby("source_id").n_test.first()
    for f in files:
        d = pd.read_csv(f)
        sid = d.source_id.iloc[0]
        ts = pd.to_datetime(d.arrival_timestamp, format="ISO8601").sort_values()
        gap = ts.diff().dt.total_seconds().div(3600).median()
        desc = d.pattern_description.iloc[0].replace("~", r"$\sim$").replace("%", r"\%").replace("_", r"\_")
        desc = desc.replace("+/-", r"$\pm$").replace("—", "--")
        mark = r"$^\dagger$" if sid in run.low_n else ""
        nt = str(int(ntest[sid])) if sid in ntest.index else "--"
        lines.append(f"{sid.replace('_', chr(92) + '_')}{mark} & {GROUP_OF.get(sid, '')} & {desc} & {len(d)} & "
                     f"{gap:.1f} & {nt} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""]
    return "\n".join(lines)


# ----------------------------------------------------------------------------- figures
def _legend_handles(models=MODELS):
    from matplotlib.lines import Line2D
    return [Line2D([], [], color=COLORS[m], marker=MARKERS[m], lw=1.5, ms=4, label=SHORT[m]) for m in models]


HIST_BINS = np.linspace(-1, 4, 101)


def fig_data_overview(data_dir: str, bts_path: str | None, out: str):
    picks = ["SRC_01", "SRC_03", "SRC_11", "SRC_13", "SRC_19", "SRC_21", "SRC_24", "SRC_25"]
    fig, axes = plt.subplots(2, 5, figsize=(7.2, 2.9))
    axes = axes.ravel()
    for ax, sid in zip(axes, picks):
        f = os.path.join(data_dir, f"{sid}.csv")
        if not os.path.exists(f):
            ax.set_visible(False)
            continue
        ts = pd.to_datetime(pd.read_csv(f).arrival_timestamp, format="ISO8601").sort_values()
        g = ts.diff().dt.total_seconds().div(3600).dropna()
        ax.hist(np.log10(g.clip(lower=0.1)), bins=HIST_BINS, color="#0072b2")
        ax.set_title(f"{sid} ({GROUP_OF.get(sid, '')})", fontsize=7)
    if bts_path and os.path.exists(bts_path):
        b = pd.read_parquet(bts_path) if bts_path.endswith(".parquet") else pd.read_csv(bts_path)
        b["ts"] = pd.to_datetime(b.arrival_timestamp, utc=True, format="ISO8601")
        g = b.sort_values("ts").groupby("source_id").ts.diff().dt.total_seconds().div(3600).dropna()
        for ax, (title, gg) in zip(axes[8:], (("BTS, all flights", g), ("BTS, gaps $\\leq$ 72 h", g[g <= 72]))):
            ax.hist(np.log10(gg.clip(lower=0.1)), bins=HIST_BINS, color="#d55e00")
            ax.set_title(title, fontsize=7)
    for ax in axes:
        ax.set_yticks([])
        ax.set_xlim(-1, 4)
        ax.set_xticks([0, 1, 2, 3])
        ax.set_xticklabels(["1h", "10h", "100h", "1kh"])
        ax.grid(False)
    fig.supxlabel("inter-arrival gap (log scale)", fontsize=8, y=0.02)
    fig.tight_layout(h_pad=0.6, w_pad=0.4)
    fig.savefig(os.path.join(out, "fig_data_overview.pdf"))
    plt.close(fig)


def fig_gain_heatmap(run: Run, out: str):
    nll, crps = run.wide("nll", valid_only=False), run.wide("crps", valid_only=False)
    order = [s for g in SOURCE_GROUPS.values() for s in g if s in nll.index]
    order += [s for s in nll.index if s not in order]
    nll, crps = nll.loc[order], crps.loc[order]
    cmp_models = ["Hawkes", "NeuralHawkes", "LogNormMix"]
    gain = -(nll[cmp_models].sub(nll["Poisson"], axis=0))
    skill = 1 - crps[cmp_models].div(crps["Poisson"], axis=0)
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 5.6))
    for ax, data, vlim, title, f in ((axes[0], gain, 4.5, "NLL gain over Poisson (nats/event)", "{:+.1f}"),
                                     (axes[1], skill, 1.0, "CRPS skill vs Poisson", "{:.2f}")):
        ax.imshow(data.clip(-vlim, vlim).to_numpy(), cmap="RdBu", vmin=-vlim, vmax=vlim, aspect="auto")
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                v = data.iat[i, j]
                txt = "< -10" if v < -10 else f.format(0.0 if abs(v) < 5e-3 else v)
                ax.text(j, i, txt, ha="center", va="center", fontsize=6,
                        color="white" if abs(min(max(v, -vlim), vlim)) > 0.6 * vlim else "black")
        ax.set_xticks(range(len(cmp_models)))
        ax.set_xticklabels([SHORT[m] for m in cmp_models])
        ax.xaxis.tick_top()
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels([s + (r"$^\dagger$" if s in run.low_n else "") for s in order])
        ax.set_title(title, pad=18)
        ax.grid(False)
        ax.tick_params(length=0)
        pos = 0
        for g, ss in SOURCE_GROUPS.items():
            k = len([s for s in ss if s in order])
            if k:
                pos += k
                if pos < len(order):
                    ax.axhline(pos - 0.5, color="black", lw=0.6)
                if ax is axes[1]:
                    ax.text(2.62, pos - k / 2 - 0.5, g, va="center", fontsize=6.5, color="#333")
    fig.tight_layout(w_pad=1.5)
    fig.savefig(os.path.join(out, "fig_gain_heatmap.pdf"))
    plt.close(fig)


def fig_calibration(runs: dict[str, Run], out: str):
    fig, axes = plt.subplots(1, len(runs), figsize=(3.4 * len(runs), 2.7), sharey=True, squeeze=False)
    levels = [0.5, 0.8, 0.9]
    for ax, (name, run) in zip(axes[0], runs.items()):
        ax.plot([0.4, 1.0], [0.4, 1.0], ls="--", color="black", lw=0.8)
        for m in MODELS:
            cov = run.wide("cov50")[m], run.wide("cov80")[m], run.wide("cov90")[m]
            med = [c.median() for c in cov]
            lo = [c.quantile(0.25) for c in cov]
            hi = [c.quantile(0.75) for c in cov]
            ax.fill_between(levels, lo, hi, color=COLORS[m], alpha=0.12, lw=0)
            ax.plot(levels, med, color=COLORS[m], marker=MARKERS[m], ms=4, label=SHORT[m])
        ax.set_xticks(levels)
        ax.set_xlabel("nominal coverage")
        ax.set_title(f"{name} ({len(run.valid)} sources)")
        ax.set_xlim(0.42, 0.98)
        ax.set_ylim(0.35, 1.02)
    axes[0][0].set_ylabel("observed coverage\n(median, IQR over sources)")
    axes[0][-1].legend(handles=_legend_handles(), loc="lower right", frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_calibration.pdf"))
    plt.close(fig)


def _per_event(run: Run) -> pd.DataFrame | None:
    if run.preds is None:
        return None
    p = run.preds[run.preds.source_id.isin(run.valid)].copy()
    return p


def fig_pit(runs: dict[str, Run], out: str):
    avail = {k: _per_event(r) for k, r in runs.items()}
    avail = {k: v for k, v in avail.items() if v is not None}
    if not avail:
        return
    fig, axes = plt.subplots(len(avail), 4, figsize=(7.2, 1.6 * len(avail) + 0.3), sharex=True, squeeze=False)
    bins = np.linspace(0, 1, 11)
    for r, (name, p) in enumerate(avail.items()):
        for c, m in enumerate(MODELS):
            ax = axes[r][c]
            g = p[p.model == m]
            # equal weight per source (and per seed within a source)
            w = 1.0 / g.groupby(["source_id"]).pit.transform("size")
            h, _ = np.histogram(g.pit.clip(0, 1), bins=bins, weights=w)
            h = h / h.sum() * len(h)
            ax.bar(bins[:-1], h, width=0.1, align="edge", color=COLORS[m], edgecolor="white", linewidth=0.5)
            ax.axhline(1, color="black", lw=0.8, ls="--")
            ax.set_ylim(0, 3.2)
            ax.grid(False)
            if r == 0:
                ax.set_title(SHORT[m])
            if c == 0:
                ax.set_ylabel(f"{name}\ndensity")
            if r == len(avail) - 1:
                ax.set_xlabel("PIT")
    fig.tight_layout(h_pad=0.4)
    fig.savefig(os.path.join(out, "fig_pit.pdf"))
    plt.close(fig)


def fig_mae_ratio(runs: dict[str, Run], out: str):
    fig, axes = plt.subplots(1, len(runs), figsize=(3.4 * len(runs), 2.6), sharey=True, squeeze=False)
    rng = np.random.default_rng(1)
    for ax, (name, run) in zip(axes[0], runs.items()):
        r = run.wide("mae_ratio")
        for i, m in enumerate(MODELS):
            v = r[m].replace([np.inf, -np.inf], np.nan).dropna().clip(1e-2, 1e4)
            ax.scatter(i + rng.uniform(-0.18, 0.18, len(v)), v, s=9, color=COLORS[m], marker=MARKERS[m],
                       alpha=0.75, lw=0)
            ax.plot([i - 0.28, i + 0.28], [v.median()] * 2, color="black", lw=1.6)
        ax.axhline(1, color="black", lw=0.8, ls="--")
        ax.set_yscale("log")
        ax.set_xticks(range(len(MODELS)))
        ax.set_xticklabels([SHORT[m] for m in MODELS])
        ax.set_title(name)
        ax.grid(axis="x", visible=False)
    axes[0][0].set_ylabel("MAE / MAE of lag-1 baseline\n(< 1 beats the baseline)")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_mae_ratio.pdf"))
    plt.close(fig)


GAP_BINS: dict[str, pd.DataFrame] = {}


def fig_gap_bins(runs: dict[str, Run], out: str):
    avail = {k: _per_event(r) for k, r in runs.items()}
    avail = {k: v for k, v in avail.items() if v is not None}
    if not avail:
        return
    bins = [0, 0.75, 1.25, 2.0, np.inf]
    labels = ["< 0.75x", "0.75-1.25x", "1.25-2x", "> 2x"]
    fig, axes = plt.subplots(1, len(avail), figsize=(3.4 * len(avail), 2.6), squeeze=False)
    for ax, (name, p) in zip(axes[0], avail.items()):
        q = p.copy()
        q["rel"] = q.y_true / q.groupby("source_id").y_true.transform("median")
        q["bin"] = pd.cut(q.rel, bins, labels=labels)
        s = q.groupby(["model", "source_id", "bin"], observed=True).nll.mean().reset_index()
        pois = s[s.model == "Poisson"].set_index(["source_id", "bin"]).nll
        s["gain"] = [pois.get((a, b), np.nan) - v for a, b, v in zip(s.source_id, s.bin, s.nll)]
        med = s.groupby(["model", "bin"], observed=True).gain.median().unstack()
        x = np.arange(len(labels))
        width = 0.26
        for i, m in enumerate(["Hawkes", "NeuralHawkes", "LogNormMix"]):
            vals = med.loc[m].reindex(labels).to_numpy() if m in med.index else np.full(len(labels), np.nan)
            ax.bar(x + (i - 1) * width, vals, width=width * 0.92, color=COLORS[m], label=SHORT[m])
        ax.axhline(0, color="black", lw=0.8)
        if np.nanmin(med.to_numpy()) < -5:  # only compress the axis when a bar dwarfs the rest
            ax.set_yscale("symlog", linthresh=1.0)
            ax.set_title(f"{name} (symlog scale)")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_xlabel("test gap relative to the source's median gap")
        n_src = q[q.model == "Poisson"].groupby("bin", observed=True).source_id.nunique().reindex(labels)
        print(f"\n{name}: median NLL gain over Poisson by relative gap size (sources per bin: "
              f"{n_src.fillna(0).astype(int).to_dict()})\n{med.reindex(columns=labels).round(2).to_string()}")
        GAP_BINS[name] = med.reindex(columns=labels)
        ax.set_title(name)
        ax.grid(axis="x", visible=False)
    axes[0][0].set_ylabel("median NLL gain over Poisson\n(nats/event)")
    axes[0][0].legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_gap_bins.pdf"))
    plt.close(fig)


def fig_example(run: Run, out: str, source: str = "SRC_19", last_n: int = 45):
    p = _per_event(run)
    if p is None or source not in set(p.source_id):
        return
    fig, axes = plt.subplots(3, 1, figsize=(7.2, 4.4), sharex=True, sharey=True)
    for ax, m in zip(axes, ["Poisson", "NeuralHawkes", "LogNormMix"]):
        g = p[(p.source_id == source) & (p.model == m)]
        if "seed" in g and g.seed.notna().any():
            g = g[g.seed == g.seed.dropna().min()]
        g = g.sort_values("arrival_timestamp").tail(last_n)
        x = pd.to_datetime(g.arrival_timestamp)
        ax.vlines(x, g.q05, g.q95, color=COLORS[m], lw=1.0, alpha=0.55, label="90% interval")
        ax.vlines(x, g.q25, g.q75, color=COLORS[m], lw=3.0, alpha=0.9, label="50% interval")
        ax.scatter(x, g.q50, marker="_", s=40, color="black", lw=1.0, label="median", zorder=3)
        ax.scatter(x, g.y_true, s=10, color="#d55e00", zorder=4, label="actual gap")
        ax.set_ylim(0, 75)
        ax.set_ylabel(f"{SHORT[m]}\ngap (h)")
        ax.grid(axis="x", visible=False)
    axes[0].legend(ncol=4, frameon=False, loc="upper left", bbox_to_anchor=(0, 1.32))
    fig.autofmt_xdate()
    fig.tight_layout(h_pad=0.3)
    fig.savefig(os.path.join(out, "fig_example_forecast.pdf"))
    plt.close(fig)


def fig_bts_gain(run: Run, out: str):
    nll, crps = run.wide("nll"), run.wide("crps")
    gain = -(nll.sub(nll["Poisson"], axis=0))
    skill = 1 - crps.div(crps["Poisson"], axis=0)
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.5))
    rng = np.random.default_rng(3)
    cm = ["Hawkes", "NeuralHawkes", "LogNormMix"]
    for ax, data, lab, lim in ((axes[0], gain, "NLL gain over Poisson (nats/event)", (-4, 6)),
                               (axes[1], skill, "CRPS skill vs Poisson", (-2, 1))):
        for i, m in enumerate(cm):
            v = data[m].dropna()
            vc = v.clip(*lim)
            ax.scatter(i + rng.uniform(-0.18, 0.18, len(v)), vc, s=9, color=COLORS[m], marker=MARKERS[m],
                       alpha=0.75, lw=0)
            ax.plot([i - 0.28, i + 0.28], [v.median()] * 2, color="black", lw=1.6)
        ax.axhline(0, color="black", lw=0.8, ls="--")
        ax.set_xticks(range(len(cm)))
        ax.set_xticklabels([SHORT[m] for m in cm])
        ax.set_ylabel(lab)
        ax.set_ylim(lim[0] - 0.1 * abs(lim[0]), lim[1] * 1.05)
        ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_bts_gain.pdf"))
    plt.close(fig)


# ----------------------------------------------------------------------------- macros
def macro_name(*parts) -> str:
    s = "".join(p[:1].upper() + p[1:] for p in parts)
    return "".join(ch for ch in s if ch.isalpha())


def numbers(prefix: str, run: Run, sb: pd.DataFrame, pt: pd.DataFrame, ts: pd.DataFrame) -> dict[str, str]:
    m = {}
    m[macro_name(prefix, "NSources")] = str(len(run.sources))
    m[macro_name(prefix, "NValid")] = str(len(run.valid))
    m[macro_name(prefix, "NLowN")] = str(len(run.low_n))
    m[macro_name(prefix, "NInformative")] = str(sb.attrs["n_informative"])
    n_test = run.agg[run.agg.model == "Poisson"].set_index("source_id").loc[run.valid].n_test.sum()
    m[macro_name(prefix, "NTestEvents")] = f"{int(n_test):,}"
    for model in MODELS:
        k = SHORT[model]
        m[macro_name(prefix, k, "Gain")] = fmt(sb.loc[model, "nll_gain"], 2)
        m[macro_name(prefix, k, "Skill")] = fmt(sb.loc[model, "crps_skill"], 2)
        m[macro_name(prefix, k, "Rank")] = fmt(sb.loc[model, "mean_rank"], 2)
        m[macro_name(prefix, k, "BestNLL")] = str(int(sb.loc[model, "best_nll"]))
        m[macro_name(prefix, k, "BestCRPS")] = str(int(sb.loc[model, "best_crps"]))
        m[macro_name(prefix, k, "CovFifty")] = fmt(sb.loc[model, "cov50"], 2)
        m[macro_name(prefix, k, "CovEighty")] = fmt(sb.loc[model, "cov80"], 2)
        m[macro_name(prefix, k, "CovNinety")] = fmt(sb.loc[model, "cov90"], 2)
        m[macro_name(prefix, k, "MaeRatio")] = fmt(sb.loc[model, "mae_ratio"], 2)
        m[macro_name(prefix, k, "BeatsLag")] = str(int(sb.loc[model, "beats_lag1"]))
        m[macro_name(prefix, k, "SlaAcc")] = fmt(sb.loc[model, "sla_acc"], 2)
        if model in ("NeuralHawkes", "LogNormMix"):
            m[macro_name(prefix, k, "Seeds")] = str(run.seeds(model))
    for _, r in pt.iterrows():
        key = macro_name(prefix, r.metric.lower(), SHORT[r.a], "vs", SHORT[r.b])
        m[key + "Wins"] = f"{r.wins}/{r.n}"
        m[key + "Effect"] = fmt(r.effect, 2 if r.metric == "NLL" else 3)
        m[key + "P"] = fmt_p(r.p)
    if ts is not None and not ts.empty:
        for model in ts.index:
            k = SHORT[model]
            m[macro_name(prefix, k, "Runs")] = str(int(ts.loc[model, "runs"]))
            m[macro_name(prefix, k, "HitCap")] = str(int(ts.loc[model, "hit_cap"]))
            m[macro_name(prefix, k, "SecPerRun")] = f"{ts.loc[model, 'med_sec']:.1f}"
            m[macro_name(prefix, k, "TotalMin")] = f"{ts.loc[model, 'total_min']:.0f}"
    ps = point_stats(run)
    for model in MODELS:
        k = SHORT[model]
        m[macro_name(prefix, k, "Mae")] = fmt(ps["mae_med"][model], 2)
        m[macro_name(prefix, k, "NllMean")] = fmt(ps["nll_mean"][model], 2)
        m[macro_name(prefix, k, "NllMed")] = fmt(ps["nll_med"][model], 2)
        m[macro_name(prefix, k, "SlaAccMed")] = f"{100 * ps['sla_acc_med'][model]:.0f}"
        m[macro_name(prefix, k, "SlaPrecMed")] = f"{100 * ps['sla_prec_med'][model]:.0f}"
        m[macro_name(prefix, k, "SlaRecMed")] = f"{100 * ps['sla_rec_med'][model]:.0f}"
    m[macro_name(prefix, "LagMae")] = fmt(ps["lag1_mae"], 2)
    m[macro_name(prefix, "MaeLogNormMixVsNHPWins")] = f"{ps['lnm_mae_wins']}/{len(run.valid)}"
    m[macro_name(prefix, "MaeLogNormMixVsNHPP")] = fmt_p(ps["mae_p"])
    m[macro_name(prefix, "NHPBigLoss")] = str(ps["nhp_big_loss"])
    m[macro_name(prefix, "MedianTestEvents")] = f"{ps['n_test_med']:.0f}"
    if run.comparison is not None and "branching_ratio" in run.comparison:
        hk = run.comparison[run.comparison.model == "Hawkes"]
        m[macro_name(prefix, "HawkesFitted")] = str(len(hk))
        m[macro_name(prefix, "HawkesDegenerate")] = str(int((hk.branching_ratio < 1e-3).sum()))
    return m


# ----------------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("synthetic", help="run folder with the final synthetic results")
    ap.add_argument("--bts", default=None, help="run folder with the frozen BTS results")
    ap.add_argument("--tuning", default="results/tuning", help="folder with tune_summary_{lnm,nhp}.csv")
    ap.add_argument("--data", default="data/synthetic", help="synthetic data folder (for the source table)")
    ap.add_argument("--bts-data", default="data/bts/DL_JFK-LAX.parquet")
    ap.add_argument("--out", default="paper/generated")
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)

    syn = Run(a.synthetic)
    bts = Run(a.bts) if a.bts else None
    runs = {"Synthetic": syn} | ({"BTS flights": bts} if bts else {})

    sb_syn, pt_syn, ts_syn = scoreboard(syn), paired_tests(syn), training_summary(syn)
    write(os.path.join(a.out, "tab_scoreboard_syn.tex"), tex_scoreboard(
        sb_syn, rf"Synthetic sources: test-set results over {len(syn.valid)} sources (medians over sources; neural "
        r"models averaged over seeds first). NLL gain = $\mathrm{NLL}_{\text{Poisson}} - \mathrm{NLL}_{\text{model}}$ "
        r"in nats/event; CRPS skill = $1-\mathrm{CRPS}/\mathrm{CRPS}_{\text{Poisson}}$; mean rank by NLL (1 = best); "
        r"MAE / lag-1 is the ratio of the median-forecast MAE to the naive last-gap forecast ($<1$ is better). "
        r"Best value per column in bold.", "tab:scoreboard-syn"))
    write(os.path.join(a.out, "tab_groups.tex"), tex_groups(by_group(syn)))
    write(os.path.join(a.out, "tab_group_mae.tex"), tex_group_mae(syn))
    sb_bts = pt_bts = ts_bts = None
    if bts:
        sb_bts, pt_bts, ts_bts = scoreboard(bts), paired_tests(bts), training_summary(bts)
        write(os.path.join(a.out, "tab_scoreboard_bts.tex"), tex_scoreboard(
            sb_bts, rf"BTS flight arrivals (held-out check, settings frozen from the synthetic study): test-set "
            rf"results over {len(bts.valid)} flight numbers. Columns as in Table~\ref{{tab:scoreboard-syn}}.",
            "tab:scoreboard-bts"))
    write(os.path.join(a.out, "tab_paired.tex"), tex_paired(pt_syn, pt_bts))
    write(os.path.join(a.out, "tab_training.tex"), tex_training(ts_syn, ts_bts))
    t = tex_tuning(a.tuning)
    if t:
        write(os.path.join(a.out, "tab_tuning.tex"), t)
    t = tex_sources(a.data, syn)
    if t:
        write(os.path.join(a.out, "tab_sources.tex"), t)

    fig_data_overview(a.data, a.bts_data, a.out)
    fig_gain_heatmap(syn, a.out)
    fig_calibration(runs, a.out)
    fig_pit(runs, a.out)
    fig_mae_ratio(runs, a.out)
    fig_gap_bins(runs, a.out)
    fig_example(syn, a.out)
    if bts:
        fig_bts_gain(bts, a.out)

    macros = numbers("syn", syn, sb_syn, pt_syn, ts_syn)
    if bts:
        macros |= numbers("bts", bts, sb_bts, pt_bts, ts_bts)
    for name, prefix in (("Synthetic", "syn"), ("BTS flights", "bts")):
        if name in GAP_BINS:
            for model, row in GAP_BINS[name].iterrows():
                for lab, key in zip(row.index, ("Short", "Typical", "Long", "VeryLong")):
                    macros[macro_name(prefix, "GapBin", SHORT[model], key)] = fmt(row[lab], 2, True)
    write(os.path.join(a.out, "numbers.tex"),
          "% Generated by `arrivalcast report` - do not edit by hand.\n" +
          "".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in sorted(macros.items())))


    pd.set_option("display.width", 200)
    print("Synthetic scoreboard:\n", sb_syn.round(3).to_string())
    if bts:
        print("\nBTS scoreboard:\n", sb_bts.round(3).to_string())
    print(f"\nwrote tables, figures and numbers.tex to {a.out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
