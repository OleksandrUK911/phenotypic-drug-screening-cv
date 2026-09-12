import numpy as np

from src.evaluation.significance import paired_bootstrap_pvalue


def test_pvalue_low_when_a_reliably_beats_b():
    rng = np.random.default_rng(0)
    scores_b = rng.normal(loc=0.70, scale=0.02, size=100)
    scores_a = scores_b + 0.10 + rng.normal(scale=0.01, size=100)

    p = paired_bootstrap_pvalue(scores_a, scores_b, n_bootstrap=1000, seed=0)
    assert p < 0.01


def test_pvalue_high_when_no_real_difference():
    rng = np.random.default_rng(0)
    scores_a = rng.normal(loc=0.70, scale=0.05, size=100)
    scores_b = rng.normal(loc=0.70, scale=0.05, size=100)

    p = paired_bootstrap_pvalue(scores_a, scores_b, n_bootstrap=1000, seed=0)
    assert p > 0.1


def test_pvalue_rejects_mismatched_lengths():
    try:
        paired_bootstrap_pvalue(np.array([1.0, 2.0]), np.array([1.0]))
    except ValueError:
        return
    raise AssertionError("Expected ValueError for mismatched lengths")
