"""
Why does the Hawkes process find (almost) no self-excitation on SRC_24?

SRC_24 is *built* to be self-exciting: one main file per day at a random time, and each main file
triggers one follow-up within four hours with probability 0.4. Across all files the true number of
follow-ups per file is therefore about 0.27, yet the fitted branching ratio alpha/beta is ~0.07.
This script tests whether that is a bug or a property of the model:

1. **Recovery check.** Simulate feeds from an exponential-kernel Hawkes process with a *constant*
   background and a known branching ratio, fit them with the project's ``HawkesExponential``, and
   check that the branching ratio is recovered.
2. **Profile likelihood on SRC_24.** Fix the branching ratio n = alpha/beta on a grid, maximise the
   training log-likelihood over (mu, beta), and see where the likelihood peaks.
3. **Mechanism.** Compare the share of short (< 4 h) gaps in SRC_24's training data with what a
   memoryless background at the fitted rate would already produce by chance.

    arrivalcast hawkes-check --out paper/generated            # ~5 min on one core

Writes ``hawkes_check.csv``, ``hawkes_profile.csv``, ``fig_hawkes_check.pdf`` and
``hawkes_numbers.tex`` (LaTeX macros) to --out.
"""
from __future__ import annotations

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.optimize import minimize  # noqa: E402

from arrivalcast.core.splits import get_chronological_windows  # noqa: E402
from arrivalcast.models.hawkes.model import HawkesExponential  # noqa: E402


def simulate_hawkes(mu: float, alpha: float, beta: float, T: float, rng: np.random.Generator) -> np.ndarray:
    """Ogata thinning with an O(1) excitation update (exponential kernel)."""
    t, excite, events = 0.0, 0.0, []
    while True:
        lam_bar = mu + excite + alpha          # upper bound until the next accepted event
        w = rng.exponential(1.0 / lam_bar)
        excite *= np.exp(-beta * w)
        t += w
        if t > T:
            return np.array(events)
        if rng.uniform() * lam_bar <= mu + excite:
            events.append(t)
            excite += alpha


def profile_loglik(gaps: np.ndarray, total_hours: float, n: float) -> float:
    """max over (mu, beta) of the log-likelihood with alpha = n * beta."""
    def nll(x):
        mu, beta = np.exp(x)
        return HawkesExponential._neg_log_likelihood(np.array([mu, n * beta, beta]), gaps, total_hours)

    best = np.inf
    for lb in (np.log(0.25), np.log(1.0), np.log(4.0)):
        x0 = np.array([np.log(len(gaps) / total_hours * (1 - n) + 1e-6), lb])
        r = minimize(nll, x0, method="Nelder-Mead", options={"xatol": 1e-6, "fatol": 1e-6, "maxiter": 2000})
        best = min(best, r.fun)
    return -best


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data/synthetic/SRC_24.csv")
    ap.add_argument("--out", default="paper/generated")
    ap.add_argument("--n-sim", type=int, default=40)
    ap.add_argument("--true-ratio", type=float, default=0.4)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)

    # ---- SRC_24 training split, exactly as the runner sees it
    ts = pd.to_datetime(pd.read_csv(a.data).arrival_timestamp, format="ISO8601").sort_values()
    days = ts.dt.normalize().value_counts().value_counts().sort_index()
    win = get_chronological_windows(ts, global_study_end=pd.Timestamp("2025-01-01"))["train"].window
    gaps, total = win.gaps_hours, win.total_hours
    fit = HawkesExponential().fit(gaps, total).params
    short_obs = float((gaps < 4).mean())
    short_bg = float(1 - np.exp(-4 * fit.mu))
    n_followups = int(days.get(2, 0))
    true_ratio_src24 = n_followups / len(ts)
    print(f"SRC_24: {len(ts)} files, days with 1/2 files = {days.to_dict()}, follow-ups per file = {true_ratio_src24:.3f}")
    print(f"fitted mu={fit.mu:.4f}/h alpha={fit.alpha:.4f} beta={fit.beta:.3f} branching={fit.branching_ratio:.3f}")
    print(f"gaps < 4h: observed {short_obs:.3f}, memoryless background at fitted mu {short_bg:.3f}")

    # ---- 1. recovery on simulated constant-background Hawkes feeds of the same size
    rng = np.random.default_rng(a.seed)
    beta_true = 0.5
    alpha_true = a.true_ratio * beta_true
    mu_true = len(gaps) / total * (1 - a.true_ratio)   # same expected event count as SRC_24's train split
    rows = []
    for k in range(a.n_sim):
        ev = simulate_hawkes(mu_true, alpha_true, beta_true, total, rng)
        p = HawkesExponential().fit(np.diff(ev), total).params
        rows.append(dict(sim=k, n_events=len(ev), mu=p.mu, alpha=p.alpha, beta=p.beta, branching_ratio=p.branching_ratio))
        print(f"  sim {k:2d}: {len(ev)} events, fitted branching ratio {p.branching_ratio:.3f}")
    sim = pd.DataFrame(rows)
    sim.to_csv(os.path.join(a.out, "hawkes_check.csv"), index=False)

    # ---- 2. profile likelihood on SRC_24
    grid = np.round(np.arange(0.0, 0.701, 0.025), 3)
    prof = pd.DataFrame({"branching_ratio": grid, "loglik": [profile_loglik(gaps, total, n) for n in grid]})
    prof["delta"] = prof.loglik - prof.loglik.max()
    prof.to_csv(os.path.join(a.out, "hawkes_profile.csv"), index=False)
    n_best = float(prof.loc[prof.loglik.idxmax(), "branching_ratio"])
    d_true = float(np.interp(true_ratio_src24, prof.branching_ratio, prof.delta))

    # ---- figure
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.5))
    ax = axes[0]
    ax.plot(prof.branching_ratio, prof.delta, color="#e69f00", marker="s", ms=3)
    ax.axvline(true_ratio_src24, color="black", ls="--", lw=0.8)
    ax.text(true_ratio_src24 + 0.01, prof.delta.min() * 0.9, f"design value\n{true_ratio_src24:.2f}", fontsize=7)
    ax.set_xlabel(r"fixed branching ratio $\alpha/\beta$")
    ax.set_ylabel("log-likelihood relative\nto the best fit")
    ax.set_title("SRC_24: profile likelihood")
    ax = axes[1]
    ax.hist(sim.branching_ratio, bins=np.linspace(0, 0.8, 33), color="#0072b2", edgecolor="white", linewidth=0.4,
            label=f"{a.n_sim} simulated feeds")
    ax.axvline(a.true_ratio, color="black", ls="--", lw=0.8, label=f"true value {a.true_ratio:.1f}")
    ax.axvline(fit.branching_ratio, color="#e69f00", lw=1.5, label=f"SRC_24 fit {fit.branching_ratio:.2f}")
    ax.set_xlabel(r"fitted branching ratio $\alpha/\beta$")
    ax.set_ylabel("feeds")
    ax.set_title("Recovery with a constant background")
    ax.legend(frameon=False, fontsize=6.5)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=0.25, lw=0.5)
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, "fig_hawkes_check.pdf"), bbox_inches="tight")
    plt.close(fig)

    m = {
        "HkFiles": f"{len(ts):,}", "HkDaysOne": str(int(days.get(1, 0))), "HkDaysTwo": str(int(days.get(2, 0))),
        "HkDaysTotal": str(int(days.sum())), "HkTrueRatio": f"{true_ratio_src24:.2f}",
        "HkFitRatio": f"{fit.branching_ratio:.2f}", "HkFitMu": f"{fit.mu:.4f}",
        "HkShortObs": f"{100 * short_obs:.1f}", "HkShortBg": f"{100 * short_bg:.1f}",
        "HkSimN": str(a.n_sim), "HkSimTrue": f"{a.true_ratio:.1f}",
        "HkSimMean": f"{sim.branching_ratio.mean():.2f}", "HkSimMin": f"{sim.branching_ratio.min():.2f}",
        "HkSimMax": f"{sim.branching_ratio.max():.2f}", "HkSimEvents": f"{int(sim.n_events.median()):,}",
        "HkProfileBest": f"{n_best:.3f}", "HkProfileDeltaTrue": f"{abs(d_true):.1f}",
    }
    with open(os.path.join(a.out, "hawkes_numbers.tex"), "w") as fh:
        fh.write("% Generated by `arrivalcast hawkes-check` - do not edit by hand.\n")
        fh.writelines(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in m.items())
    print(f"\nrecovery: mean {sim.branching_ratio.mean():.3f} (range {sim.branching_ratio.min():.3f}-"
          f"{sim.branching_ratio.max():.3f}); profile peaks at {n_best:.3f}; at the design value the "
          f"log-likelihood is {d_true:.1f} below the best")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
