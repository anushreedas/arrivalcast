# Methodology notes

## Task
For each source, independently: given the arrival history up to event *i*, predict the
distribution of the gap Δt<sub>i+1</sub> until the next arrival.

## Protocol
1. **Split.** Split each source chronologically by event count: 70% train, 15% validation,
   15% test.
2. **Censoring.** A split's observation window ends at the first event of the next split. The
   test split ends at the dataset's real study end (2025-01-01 for both datasets). The quiet
   tail after the last event enters the likelihood as a survival term.
3. **Training.**
   - The neural models use truncated BPTT over windows of about 128 events, each preceded by
     32 burn-in events that carry no loss. They train with Adam and gradient clipping, and stop
     early on validation NLL (patience 20).
   - Sequences shorter than 2 × 128 events use the exact full-sequence likelihood, including
     the censored tail.
4. **Scoring.** Test events are scored with the model state replayed through all earlier events.
   The canonical metric is the mean NLL over completed test gaps, which is identical for all
   four models.
5. **Exclusions.** Sources with fewer than 5 test events (`low_n`) are reported but left out of
   headline statistics.
6. **Model selection.** Hyper-parameters are chosen on validation NLL only. A configuration
   replaces the reference only if its mean improvement is more than 2 standard errors.
   Result: LogNormMix learning rate 3e-3; Neural Hawkes stays at 1e-2.
7. **External check.** BTS flights are run once with the frozen settings.

## Metrics
| Metric | Definition | Why |
|---|---|---|
| NLL | −log p(Δt) in nats (time in hours) | proper scoring rule, sensitive to sharpness and to tail failures |
| CRPS | ∫(F(t) − 1[t ≥ y])² dt by adaptive quadrature | proper, in hours, robust to tail misspecification |
| PICP | coverage of central 50/80/90% intervals | calibration |
| SLA-hit | P(Δt ≤ 24 h) ≥ 0.5 vs. actual | the operational yes/no question |
| MAE / lag-1 | MAE of the median forecast ÷ MAE of "same gap as last time" | a non-statistical floor |

Cross-source summaries use the NLL gain over Poisson and the CRPS skill score, because raw NLL
and CRPS scale with each source's cadence.
