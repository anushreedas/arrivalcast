import numpy as np
import pytest

from arrivalcast.evaluation.baseline import lag1_predictions
from arrivalcast.evaluation.metrics import (
    crps_from_cdf,
    pct_improvement,
    picp,
    quantile_from_cdf,
    sla_hit_classification,
)


def exp_cdf(rate):
    return lambda t: 1.0 - np.exp(-rate * np.asarray(t, dtype=float))


def exp_crps(rate, y):
    # closed form CRPS of Exponential(rate) at observation y
    return y + 2.0 / rate * np.exp(-rate * y) - 1.5 / rate


@pytest.mark.parametrize("rate,y", [(1.0, 0.3), (1.0, 4.0), (1 / 24, 23.0), (1 / 2000, 10.0), (50.0, 0.01)])
def test_crps_matches_closed_form_exponential(rate, y):
    assert crps_from_cdf(exp_cdf(rate), y) == pytest.approx(exp_crps(rate, y), rel=2e-3)


def test_crps_point_mass_limit_is_absolute_error():
    # a very sharp distribution around 5 -> CRPS ~ |y - 5|
    from scipy.stats import norm
    cdf = lambda t: norm.cdf(np.asarray(t), loc=5.0, scale=1e-3)
    assert crps_from_cdf(cdf, 8.0) == pytest.approx(3.0, rel=1e-3)


@pytest.mark.parametrize("p", [0.05, 0.5, 0.9])
@pytest.mark.parametrize("rate", [1.0, 1 / 24, 1 / 5000])
def test_quantile_inverts_exponential(p, rate):
    q = quantile_from_cdf(exp_cdf(rate), p)
    assert q == pytest.approx(-np.log(1 - p) / rate, rel=1e-6)


def test_picp_counts_inclusive_bounds():
    r = picp(np.array([0, 1, 2]), np.array([1, 2, 3]), np.array([1, 5, 2]), nominal_level=0.8)
    assert r.n_covered == 2 and r.observed_coverage == pytest.approx(2 / 3)


def test_sla_classification_and_f1_edge_cases():
    r = sla_hit_classification(np.array([1, 1, 0, 0], bool), np.array([1, 0, 1, 0], bool))
    assert (r.tp, r.fp, r.fn, r.tn) == (1, 1, 1, 1)
    assert r.f1 == pytest.approx(0.5)
    none = sla_hit_classification(np.zeros(3, bool), np.zeros(3, bool))
    assert np.isnan(none.f1) and none.accuracy == 1.0
    miss = sla_hit_classification(np.zeros(2, bool), np.ones(2, bool))
    assert miss.f1 == 0.0


def test_lag1_baseline_and_pct_improvement():
    pred = lag1_predictions(np.array([1.0, 2.0, 3.0]))
    assert np.isnan(pred[0]) and np.allclose(pred[1:], [1.0, 2.0])
    assert pct_improvement(10.0, 5.0) == pytest.approx(50.0)
