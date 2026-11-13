import numpy as np
import pytest
import torch
from scipy.integrate import quad

from arrivalcast.features.calendar_features import CALENDAR_FEATURE_NAMES, compute_calendar_features
from arrivalcast.features.time2vec import RawGapEncoding, Time2Vec
from arrivalcast.models.lognormmix.model import LogNormMix
from arrivalcast.models.lognormmix.predictive import LogNormMixPredictive
from arrivalcast.models.neural_hawkes.model import CTLSTMCell, NeuralHawkes
from arrivalcast.models.neural_hawkes.predictive import NeuralHawkesPredictive


def test_calendar_features_are_cyclical_and_named(daily_timestamps):
    f = compute_calendar_features(daily_timestamps)
    assert list(f.columns) == CALENDAR_FEATURE_NAMES
    assert np.allclose(f.hour_sin**2 + f.hour_cos**2, 1.0)
    assert f.notna().all().all()


@pytest.mark.parametrize("shape,expected", [((1,), (1, 8)), ((5,), (5, 8)), ((2, 7, 1), (2, 7, 8))])
def test_time2vec_output_shape(shape, expected):
    # 0-D/1-D inputs get a feature axis; higher-rank inputs must already end in a size-1 axis
    assert Time2Vec(out_dim=8)(torch.rand(*shape)).shape == expected
    assert RawGapEncoding()(torch.rand(4)).shape == (4, 1)


def test_lognormmix_torch_and_numpy_densities_agree():
    torch.manual_seed(0)
    m = LogNormMix(input_dim=3, hidden_dim=8, n_components=3).eval()
    h = torch.randn(1, 8)
    y = torch.tensor([5.0])
    lp_torch = m.decoder.log_prob(h, y).item()
    lw, mu, ls = m.decoder.get_params(h)
    pred = LogNormMixPredictive(lw.exp()[0].detach().numpy(), mu[0].detach().numpy(), ls.exp()[0].detach().numpy())
    assert pred.log_prob(5.0) == pytest.approx(lp_torch, abs=1e-4)
    total, _ = quad(lambda t: np.exp(pred.log_prob(t)), 0, np.inf, limit=400)
    assert total == pytest.approx(1.0, abs=1e-4)
    # survival in torch == 1 - numpy cdf
    s_torch = m.decoder.log_survival(h, torch.tensor([7.0])).exp().item()
    assert s_torch == pytest.approx(1 - pred.cdf(np.array([7.0]))[0], abs=1e-5)


def test_lognormmix_likelihood_is_finite_and_differentiable():
    m = LogNormMix(input_dim=1, hidden_dim=8, n_components=2)
    x = torch.randn(2, 6, 1)
    ll = m.log_likelihood(x, torch.rand(2, 6) + 0.1, torch.ones(2, 6), torch.tensor([6, 6]),
                          torch.tensor([1.0, 2.0]), torch.randn(2, 1))
    assert ll.shape == (2,) and torch.isfinite(ll).all()
    (-ll.sum()).backward()
    assert all(p.grad is not None for p in m.parameters() if p.requires_grad)


def test_ctlstm_decay_endpoints():
    c, cb, d = torch.tensor([2.0]), torch.tensor([-1.0]), torch.tensor([0.5])
    assert CTLSTMCell.decay_cell(c, cb, d, torch.tensor([0.0])).item() == pytest.approx(2.0)
    assert CTLSTMCell.decay_cell(c, cb, d, torch.tensor([1e4])).item() == pytest.approx(-1.0)


def test_neural_hawkes_forward_and_predictive():
    torch.manual_seed(0)
    nhp = NeuralHawkes(input_dim=4, hidden_dim=8).eval()
    B, L, K = 2, 5, 3
    gaps = torch.rand(B, L) + 0.5
    integ = torch.rand(B, L, K) * gaps.unsqueeze(-1)
    log_int, comp, final, states = nhp.forward(torch.randn(B, L, 4), gaps, integ, lengths=torch.tensor([5, 3]))
    assert log_int.shape == comp.shape == (B, L)
    assert (comp >= 0).all()
    assert states[0].shape == (B, L, 8)
    # final state of the shorter sequence is the state after its 3rd event
    assert torch.allclose(final[0][1], states[0][1, 2])
    state = tuple(s[0, -1] for s in states)
    pred = NeuralHawkesPredictive(nhp, state, n_quad_points=2000)
    s = pred.survival(np.array([0.0, 1.0, 10.0]))
    assert s[0] == pytest.approx(1.0) and s[0] >= s[1] >= s[2]
