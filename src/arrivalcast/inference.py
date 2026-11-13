"""
Production inference: "when will this source's next file arrive?"

Loads a trained LogNormMix checkpoint (written by the experiment runner to
``results/<run>/models/<source>__lnm_s<seed>.pt``) and turns a source's
arrival history into a full predictive distribution for the next arrival:

* quantiles of the next inter-arrival gap and the matching wall-clock times,
* P(next arrival within an SLA window), and
* the same probability *conditioned on the file not having arrived yet*
  ("it is now 10:30 and the 09:00 file is still missing - what is the chance
  it lands before the 12:00 deadline?"), which is what an alerting system
  actually needs.

The history is replayed through the GRU exactly as in evaluation
(:mod:`arrivalcast.models.lognormmix.evaluate`), so the state used for
the forecast has consumed every observed gap, including the last one.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
import torch

from arrivalcast.evaluation.metrics import quantile_from_cdf
from arrivalcast.features.calendar_features import compute_calendar_features
from arrivalcast.models.lognormmix.dataloader import (
    LogNormMixSequence,
    build_step_inputs,
    build_terminal_input,
)
from arrivalcast.models.lognormmix.model import LogNormMix
from arrivalcast.models.lognormmix.predictive import LogNormMixPredictive
from arrivalcast.training import TrainConfig, load_lnm

DEFAULT_QUANTILES = (0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95)


@dataclass
class NextArrivalForecast:
    last_arrival: pd.Timestamp
    quantiles_hours: dict[float, float]
    quantile_timestamps: dict[float, pd.Timestamp]
    mean_gap_hours: float
    sla_hours: float | None = None
    p_within_sla: float | None = None
    as_of: pd.Timestamp | None = None
    p_within_sla_given_not_yet: float | None = None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["quantile_timestamps"] = {k: str(v) for k, v in self.quantile_timestamps.items()}
        for k in ("last_arrival", "as_of"):
            d[k] = None if d[k] is None else str(d[k])
        return d


class LogNormMixForecaster:
    """Next-arrival forecaster for one source, backed by a trained LogNormMix."""

    def __init__(self, model: LogNormMix, cfg: TrainConfig):
        self.model = model.eval()
        self.cfg = cfg

    @classmethod
    def from_checkpoint(cls, path: str) -> LogNormMixForecaster:
        model, cfg = load_lnm(path)
        return cls(model, cfg)

    @torch.no_grad()
    def predictive(self, timestamps: pd.Series) -> LogNormMixPredictive:
        """Predictive distribution of the gap after the last timestamp."""
        ts = pd.to_datetime(pd.Series(timestamps)).sort_values().reset_index(drop=True)
        if len(ts) < 3:
            raise ValueError("need at least 3 arrivals (2 gaps) of history to forecast")
        gaps = ts.diff().dropna().dt.total_seconds().to_numpy() / 3600.0
        if self.cfg.use_calendar_features:
            cal = compute_calendar_features(ts).to_numpy()[1:]
        else:
            cal = np.zeros((len(gaps), 0))
        seq = LogNormMixSequence(gaps_hours=gaps, calendar_features=cal, tail_hours=0.0, n_gaps=len(gaps))
        inputs, _ = build_step_inputs(seq)
        x = torch.tensor(inputs, dtype=torch.float32).unsqueeze(0)
        terminal = torch.tensor(build_terminal_input(seq), dtype=torch.float32).unsqueeze(0)
        lengths = torch.tensor([inputs.shape[0]], dtype=torch.long)
        h = self.model._state_after_last_gap(self.model.forward(x), lengths, terminal)
        log_w, means, log_stds = self.model.decoder.get_params(h)
        return LogNormMixPredictive(
            weights=log_w.exp().squeeze(0).numpy(),
            means=means.squeeze(0).numpy(),
            stds=log_stds.exp().squeeze(0).numpy(),
        )

    def forecast(
        self,
        timestamps: pd.Series,
        quantiles: tuple[float, ...] = DEFAULT_QUANTILES,
        sla_hours: float | None = None,
        as_of: pd.Timestamp | None = None,
    ) -> NextArrivalForecast:
        """
        sla_hours : optional deadline, measured from the last arrival.
        as_of : optional "now". If given (and later than the last arrival),
            the SLA probability is also reported conditional on no arrival
            having happened between the last arrival and ``as_of``.
        """
        ts = pd.to_datetime(pd.Series(timestamps)).sort_values().reset_index(drop=True)
        pred = self.predictive(ts)
        last = ts.iloc[-1]
        hint = pred.suggested_t_max()
        q_hours = {float(q): quantile_from_cdf(pred.cdf, q, hint) for q in quantiles}
        q_ts = {q: last + pd.Timedelta(hours=h) for q, h in q_hours.items()}
        out = NextArrivalForecast(
            last_arrival=last, quantiles_hours=q_hours, quantile_timestamps=q_ts,
            mean_gap_hours=pred.mean(),
        )
        if sla_hours is not None:
            out.sla_hours = float(sla_hours)
            out.p_within_sla = float(pred.cdf(np.array([sla_hours]))[0])
            if as_of is not None:
                as_of = pd.Timestamp(as_of)
                elapsed = (as_of - last).total_seconds() / 3600.0
                out.as_of = as_of
                out.p_within_sla_given_not_yet = conditional_arrival_probability(pred, elapsed, sla_hours)
        return out


def conditional_arrival_probability(pred, elapsed_hours: float, deadline_hours: float) -> float:
    """P(gap <= deadline | gap > elapsed) = (F(d) - F(e)) / (1 - F(e)).

    Returns 0 if the deadline has already passed, and 1.0 - tiny if the
    survival at ``elapsed`` underflows (the model considers the arrival
    overdue beyond anything seen in training)."""
    if elapsed_hours <= 0:
        return float(pred.cdf(np.array([deadline_hours]))[0])
    if deadline_hours <= elapsed_hours:
        return 0.0
    f_e, f_d = pred.cdf(np.array([elapsed_hours, deadline_hours]))
    surv = 1.0 - f_e
    if surv <= 1e-12:
        return float("nan")
    return float(np.clip((f_d - f_e) / surv, 0.0, 1.0))
