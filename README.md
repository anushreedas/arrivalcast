# ArrivalCast: when will the next file arrive?

[![CI](https://github.com/anushreedas/arrivalcast/actions/workflows/ci.yml/badge.svg)](https://github.com/anushreedas/arrivalcast/actions/workflows/ci.yml)
![python](https://img.shields.io/badge/python-3.10%20%7C%203.11-blue)
![license](https://img.shields.io/badge/license-MIT-green)

ArrivalCast produces **probabilistic forecasts of the next arrival time** of recurring events.
Examples are a daily vendor file, an hourly log drop, or a scheduled flight. Each forecast is a
full predictive distribution: quantiles, prediction intervals, and the probability that the event
arrives before an SLA deadline. That probability can also be updated while the event is late
("it is 10:30 and the 09:00 file is still missing").

The repository compares four temporal point-process models under one leakage-safe protocol:

| Model | Type | History / covariates | Predictive CDF |
|---|---|---|---|
| Homogeneous **Poisson** | classical | none (memoryless) | closed form |
| Exponential-kernel **Hawkes** | classical, self-exciting | excitation state *R* | closed form |
| **Neural Hawkes** (CT-LSTM, Mei & Eisner 2017) | neural, intensity-based | Time2Vec + calendar features | numerical quadrature |
| **LogNormMix** (Shchur et al. 2020) | neural, intensity-free | log-gap + calendar features | closed form (mixture of log-normals) |

It is the code for my master's capstone. The paper is in [`paper/`](paper/), and every number,
table and figure in it is generated from the result files by `arrivalcast report`.

## Headline results

Test-set medians over sources. Neural models are averaged over seeds first (LogNormMix 10 seeds,
Neural Hawkes 5). The NLL gain is in nats/event over Poisson; the CRPS skill is
`1 - CRPS/CRPS_Poisson`.

| | Synthetic (21 sources) | | | BTS flights, held-out (34 flight numbers) | | |
|---|---|---|---|---|---|---|
| Model | NLL gain | CRPS skill | 80% PI coverage | NLL gain | CRPS skill | 80% PI coverage |
| Poisson | 0.00 | 0.00 | 1.00 | 0.00 | 0.00 | 1.00 |
| Hawkes | 0.00 | 0.00 | 1.00 | 0.00 | 0.00 | 1.00 |
| Neural Hawkes | +1.76 | **0.80** | 1.00 | +0.55 | 0.46 | 0.93 |
| **LogNormMix** | **+3.31** | 0.76 | **0.81** | **+2.35** | **0.69** | 0.71 |

* **LogNormMix has the lowest NLL on 20 of 21 synthetic sources and 29 of 34 flights.** It is
  the only model whose intervals are close to calibrated. The other models' 80% intervals cover
  almost every outcome, which means they are much too wide.
* Neural Hawkes is competitive on CRPS for synthetic data (it wins 10 of 21 sources). On real
  flight data its NLL advantage over Poisson is not significant (it is better on 22 of 34
  flights, p = 0.57). The reason is that it fails badly on unusually long gaps: median
  −34.9 nats/event on gaps longer than 2× the typical gap. Each training run also takes about
  9× longer than a LogNormMix run.
* Hawkes collapses to Poisson (branching ratio < 10⁻³) on 21 of 24 synthetic sources and on all
  41 fitted flights. It even misses most of the follow-up files on the source that was *built* to
  be self-exciting: it fits a branching ratio of 0.07 where the design value is 0.27.
  `arrivalcast hawkes-check` shows this is a model mismatch, not a bug:
  - on 40 simulated feeds with a random background, the same code recovers the true ratio (mean 0.40);
  - the profile likelihood peaks near 0.075;
  - a constant background cannot represent a daily schedule, so it absorbs the short gaps.
* On point forecasts the neural models trade places by schedule type. Neural Hawkes is sharper on
  calendar-driven feeds (median MAE 0.50 h vs 4.22 h). LogNormMix is far better on low-frequency
  feeds, where simply repeating the last gap beats both. On real flights LogNormMix wins
  outright: median MAE 2.95 h vs 4.55 h for Neural Hawkes, 8.75 h for Poisson and 6.74 h for the
  lag-1 rule.
* The BTS evaluation is a **held-out check**. Settings are frozen from the synthetic study, and
  the CLI refuses to tune on BTS.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"        # add ,notebooks for the analysis notebooks
make test                      # 51 tests: unit + end-to-end smoke training (~1 min on CPU)
```

```bash
# fit + evaluate all four models on two sources, a few seconds each
arrivalcast run --run-name demo --sources SRC_01 SRC_19 --seeds 0 --lnm-lr 0.003

# forecast the next arrival from a history file with the trained LogNormMix
arrivalcast predict --checkpoint results/demo/models/SRC_19__lnm_s0.pt \
    --history data/synthetic/SRC_19.csv --sla-hours 26 --as-of "2025-01-01 12:00"
```

`predict` returns JSON with the next gap's quantiles (in hours and as timestamps), P(arrival ≤ SLA),
and the same probability conditioned on the file not having arrived by `--as-of`.

To use the forecaster from Python:

```python
from arrivalcast.inference import LogNormMixForecaster
fc = LogNormMixForecaster.from_checkpoint("results/demo/models/SRC_19__lnm_s0.pt")
print(fc.forecast(history_timestamps, sla_hours=26).to_dict())
```

## Reproducing the study

| Step | Command | Cost (1 CPU core) |
|---|---|---|
| Regenerate synthetic data (deterministic, seed 42) | `make data` | seconds |
| Extract BTS flights (needs the monthly BTS ZIPs or `--mode download`) | `arrivalcast data bts --route JFK-LAX-DL` | minutes |
| Final synthetic study | `make reproduce-syn` | ~2.5 h (Neural Hawkes dominates) |
| Hyper-parameter selection (validation loss only) | `arrivalcast tune-summary ...` | see `results/tuning/` |
| Held-out BTS check, settings frozen | `make reproduce-bts` | ~45 min |
| Paper tables, figures and numbers (incl. the Hawkes diagnostic) | `make report` | ~5 min |
| Paper PDF | `make paper` | ~10 s |

Runs are resumable. Each `(source, model, seed)` unit is cached in `results/<run>/parts/`, so
adding seeds only trains the new ones.

## Engineering decisions that keep the evaluation honest

* **Chronological 70/15/15 split per source.** The split is by event count, so sparse monthly
  and quarterly sources still get a test set. Model state (Hawkes *R*, GRU/CT-LSTM hidden state)
  is **replayed through train and validation** before scoring test events instead of being reset.
* **Right-censoring is modelled.** Each split's likelihood includes the survival term up to the
  next split's first event, or up to the study end. The study end is the real extraction cutoff,
  never the last observed event.
* **One scoring path for every model.** Each model exposes only `cdf()` and `log_prob()`.
  NLL, CRPS (adaptive log-spaced quadrature, tested against the closed form), PICP, SLA-hit and
  MAE-vs-lag-1 are implemented once in
  [`evaluation/`](src/arrivalcast/evaluation/), so no model gets a more precise estimator than
  another.
* **Guards against accidental test-set tuning.**
  - BTS runs require `--freeze-from <synthetic run>`.
  - A frozen run rejects any training flag.
  - Reusing a run name with different settings is refused.
  - Default run names are hashes of the configuration.
* **Hyper-parameters are chosen on validation loss only**, with a 2-SE rule against a reference
  (`arrivalcast tune-summary`).
* **Reproducibility.** Seeded per-source data generation, seeded training, single-threaded torch,
  checkpoints that store their config, and a pinned `requirements-lock.txt`.

## Repository layout

```
src/arrivalcast/
  data/          io.py (loader, one schema for both datasets), pipeline.py (synthetic generator + BTS extractor)
  core/          splits.py (chronological splits + censoring bounds), windows.py (observation windows)
  features/      calendar_features.py (cyclical calendar covariates), time2vec.py
  models/        poisson/ hawkes/ neural_hawkes/ lognormmix/   (model, dataloader, predictive, evaluate)
  evaluation/    metrics.py (CRPS, quantiles, PICP, SLA), evaluate.py (per-source aggregation), baseline.py
  training.py    truncated-BPTT training loop, early stopping, checkpoints
  inference.py   LogNormMixForecaster: next-arrival quantiles and (conditional) SLA probabilities
  experiments/   runner.py, tune_summary.py, compare_runs.py, report.py, hawkes_check.py
  cli.py         `arrivalcast {data,run,tune-summary,compare,report,hawkes-check,predict}`
tests/           51 tests: closed-form checks, brute-force likelihoods, determinism, CLI end-to-end
notebooks/       EDA, initial experiments, results analysis (read-only over results/)
results/         run summaries (CSV/JSON/figures); checkpoints and per-event dumps are git-ignored
paper/           LaTeX capstone paper; paper/generated/ is written by `arrivalcast report`
docs/            methodology notes, model card
```

## Data

* **Synthetic (26 sources, 2023–2024).** Daily, weekday, monthly, quarterly and bursty
  schedules; outages; seasonal jitter; size-dependent delays; a Hawkes-like source; and a Pareto
  heavy-tailed source. They are generated by `arrivalcast data synthetic`, and each source is
  seeded from its own id. See [`data/README.md`](data/README.md).
* **BTS On-Time Performance (Delta, JFK→LAX, 2023–2024).** Each flight number's actual arrivals
  form one source (56 flight numbers, 6,527 arrivals). The data is public US DOT data; the raw
  monthly ZIPs (~660 MB) are not committed.

## Paper

[`paper/main.tex`](paper/main.tex) is the capstone report. Build it with `make paper`;
CI also builds it and uploads the PDF as an artifact.

## License

MIT. See [LICENSE](LICENSE).
