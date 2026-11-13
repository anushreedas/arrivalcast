# Model card: LogNormMix next-arrival forecaster

**Intended use.** Monitor recurring data deliveries or scheduled events. The model gives a
calibrated distribution of the time to the next arrival, and the probability that an overdue
event still arrives before a deadline.

**Architecture.**
- A single-layer GRU (hidden size 32) reads, at each step:
  - the standardised log inter-arrival gap, and
  - 9 calendar covariates of the arrival (hour, day of week and month as sin/cos pairs;
    weekend; business hours; US federal holiday).
- The hidden state feeds a mixture of 4 log-normals over the next gap, with σ ≤ e.
- The model is trained per source (about 4.6k parameters).

**Training data.** The arrival history of the source itself. In this study: 26 synthetic
schedules (2023–2024) and 56 Delta JFK→LAX flight numbers (BTS, 2023–2024).

**Evaluation.** Chronological hold-out (last 15% of events) with the state replayed through all
earlier events. See the README and paper for results.

**Limitations.**
- Sources with fewer than about 30 events are unreliable; seed variance is large there.
- The model does not know about announced schedule changes. On flight data, a long service
  pause (≫ the typical gap) is the main failure mode.
- Calendar features assume US holidays and UTC timestamps.
- It is a per-source model, with no pooling across sources.

**Monitoring suggestions.** Track the PIT histogram and the 80% interval coverage on a rolling
window. Retrain when the coverage leaves [0.7, 0.9].
