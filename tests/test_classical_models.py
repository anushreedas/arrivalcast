import numpy as np
import pytest
from scipy.integrate import quad

from arrivalcast.models.hawkes.model import HawkesExponential, HawkesParams
from arrivalcast.models.hawkes.predictive import HawkesPredictive, compute_R_trajectory
from arrivalcast.models.poisson.model import HomogeneousPoisson
from arrivalcast.models.poisson.predictive import PoissonPredictive


def brute_force_hawkes_ll(mu, alpha, beta, times, T):
    ll = 0.0
    for i, t in enumerate(times):
        ll += np.log(mu + alpha * np.sum(np.exp(-beta * (t - times[:i]))))
    comp = mu * T + alpha / beta * np.sum(1 - np.exp(-beta * (T - times)))
    return ll - comp


def simulate_hawkes(mu, alpha, beta, T, seed=0):
    """Ogata thinning."""
    rng = np.random.default_rng(seed)
    t, events = 0.0, []
    while True:
        lam_bar = mu + alpha * np.sum(np.exp(-beta * (t - np.array(events)))) + alpha
        t += rng.exponential(1 / lam_bar)
        if t > T:
            return np.array(events)
        lam_t = mu + alpha * np.sum(np.exp(-beta * (t - np.array(events))))
        if rng.uniform() <= lam_t / lam_bar:
            events.append(t)


def test_poisson_mle_and_predictive():
    m = HomogeneousPoisson().fit(n_events=48, total_hours=96.0)
    assert m.rate_per_hour == pytest.approx(0.5)
    p = PoissonPredictive(m)
    assert p.median() == pytest.approx(np.log(2) / 0.5)
    assert p.cdf(np.array([p.median()]))[0] == pytest.approx(0.5)


def test_hawkes_recursive_likelihood_matches_brute_force():
    times = np.array([0.0, 0.5, 0.7, 3.0, 3.1, 3.15, 8.0])
    T = 10.0
    mu, alpha, beta = 0.3, 0.8, 1.7
    ll = -HawkesExponential._neg_log_likelihood(np.array([mu, alpha, beta]), np.diff(times), T)
    assert ll == pytest.approx(brute_force_hawkes_ll(mu, alpha, beta, times, T), rel=1e-10)


def test_hawkes_recovers_self_excitation():
    times = simulate_hawkes(mu=0.5, alpha=0.6, beta=1.5, T=600.0, seed=1)
    m = HawkesExponential().fit(np.diff(times), total_hours=600.0)
    assert m.params.branching_ratio == pytest.approx(0.4, abs=0.15)
    assert m.params.mu == pytest.approx(0.5, rel=0.35)


def test_hawkes_predictive_density_integrates_to_one():
    pred = HawkesPredictive(HawkesParams(mu=0.2, alpha=1.0, beta=2.0), R=1.5)
    total, _ = quad(lambda s: np.exp(pred.log_prob(s)), 0, np.inf, limit=200)
    assert total == pytest.approx(1.0, abs=1e-6)


def test_hawkes_state_includes_conditioning_event():
    R_before, R_final = compute_R_trajectory(np.array([1.0, 1.0]), beta=1.0)
    assert R_before[0] == pytest.approx(1.0)              # just the event itself
    assert R_before[1] == pytest.approx(1.0 + np.exp(-1))  # plus the decayed earlier one
    assert R_final == pytest.approx(np.exp(-1) * (1 + np.exp(-1)))


def test_hawkes_check_simulator_and_profile():
    from arrivalcast.experiments.hawkes_check import profile_loglik, simulate_hawkes

    rng = np.random.default_rng(3)
    ev = simulate_hawkes(mu=0.2, alpha=0.25, beta=0.5, T=2000.0, rng=rng)
    # stationary rate mu / (1 - n) = 0.2 / 0.5 = 0.4 events per hour
    assert len(ev) / 2000.0 == pytest.approx(0.4, rel=0.15)
    assert np.all(np.diff(ev) > 0)
    # the profile likelihood is maximised near the true branching ratio
    gaps = np.diff(ev[:400])
    total = float(ev[399])
    ll = {n: profile_loglik(gaps, total, n) for n in (0.0, 0.5, 0.9)}
    assert ll[0.5] > ll[0.0] and ll[0.5] > ll[0.9]
