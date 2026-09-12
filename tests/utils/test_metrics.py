import numpy as np

from src.utils.metrics import bootstrap_ci


def test_bootstrap_ci_degenerate_for_constant_values():
    values = np.full(50, 7.0)
    lower, upper = bootstrap_ci(values, n_bootstrap=200)
    assert lower == upper == 7.0


def test_bootstrap_ci_contains_true_mean_for_normal_sample():
    rng = np.random.default_rng(1)
    values = rng.normal(loc=10.0, scale=1.0, size=500)

    lower, upper = bootstrap_ci(values, n_bootstrap=1000, ci=0.95, seed=1)
    assert lower < 10.0 < upper


def test_bootstrap_ci_narrower_with_more_data():
    rng = np.random.default_rng(2)
    small = rng.normal(loc=0.0, scale=1.0, size=20)
    large = rng.normal(loc=0.0, scale=1.0, size=2000)

    lo_s, hi_s = bootstrap_ci(small, n_bootstrap=500, seed=2)
    lo_l, hi_l = bootstrap_ci(large, n_bootstrap=500, seed=2)
    assert (hi_l - lo_l) < (hi_s - lo_s)
