import dataclasses

import numpy as np
import pandas as pd
import pytest
import torch

from arrivalcast import training as T
from arrivalcast.inference import LogNormMixForecaster, conditional_arrival_probability
from arrivalcast.models.lognormmix.evaluate import evaluate_lognormmix_source
from arrivalcast.models.poisson.model import HomogeneousPoisson
from arrivalcast.models.poisson.predictive import PoissonPredictive

FAST = dataclasses.replace(T.DEFAULT_CONFIG, max_epochs=3, patience=2, hidden_dim=8, n_components=2)
END = pd.Timestamp("2023-06-01")


def test_make_spans_cover_every_position_once():
    spans = T.make_spans(1000, chunk_len=128, burn_in=32)
    loss_positions = np.concatenate([np.arange(s, b) for _, s, b in spans])
    assert np.array_equal(loss_positions, np.arange(1000))
    assert all(a == max(s - 32, 0) for a, s, _ in spans)
    assert T.make_spans(100, 128, 32) == [(0, 0, 100)]


def test_config_id_ignores_seed():
    a = dataclasses.replace(T.DEFAULT_CONFIG, seed=1)
    assert a.config_id() == T.DEFAULT_CONFIG.config_id()
    assert dataclasses.replace(a, lr=1e-3).config_id() != a.config_id()


@pytest.mark.slow
def test_lnm_train_is_deterministic_and_checkpoint_roundtrips(tmp_path, daily_timestamps):
    df = pd.DataFrame({"arrival_timestamp": daily_timestamps})
    r1 = T.train_lnm_one_source(df, END, FAST)
    r2 = T.train_lnm_one_source(df, END, FAST)
    assert r1 is not None and np.isfinite(r1[0])
    assert r1[0] == pytest.approx(r2[0], abs=1e-6)  # same seed -> same model
    model, info = r1[2], r1[3]
    path = tmp_path / "m.pt"
    T.save_checkpoint(path, "lnm", FAST, info["_input_dim"], model=model)
    loaded, cfg = T.load_lnm(path)
    assert cfg == FAST
    for a, b in zip(model.state_dict().values(), loaded.state_dict().values()):
        assert torch.equal(a, b)
    m = evaluate_lognormmix_source("S", daily_timestamps, END, loaded)
    assert m.n_test_events > 0 and np.isfinite(m.mean_nll) and np.isfinite(m.mean_crps)

    fc = LogNormMixForecaster(loaded, cfg).forecast(daily_timestamps, sla_hours=24.0,
                                                    as_of=daily_timestamps.iloc[-1] + pd.Timedelta(hours=20))
    qs = list(fc.quantiles_hours.values())
    assert all(np.diff(qs) > 0)                       # quantiles are increasing
    assert 0.0 <= fc.p_within_sla <= 1.0
    assert 0.0 <= fc.p_within_sla_given_not_yet <= 1.0


@pytest.mark.slow
def test_nhp_trains_and_reports_finite_likelihood(daily_timestamps):
    df = pd.DataFrame({"arrival_timestamp": daily_timestamps})
    res = T.train_nhp_one_source(df, END, dataclasses.replace(FAST, max_epochs=2))
    assert res is not None
    test_ll, n_test, model, t2v, info = res
    assert np.isfinite(test_ll) and n_test > 0 and info["epochs_run"] >= 1


def test_conditional_arrival_probability():
    pred = PoissonPredictive(HomogeneousPoisson(rate_per_hour=0.1))
    # memoryless: P(arrive within next 5h | waited 10h) == P(arrive within 5h)
    assert conditional_arrival_probability(pred, 10.0, 15.0) == pytest.approx(1 - np.exp(-0.5))
    assert conditional_arrival_probability(pred, 10.0, 9.0) == 0.0
