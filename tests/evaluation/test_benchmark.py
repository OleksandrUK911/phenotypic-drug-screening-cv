import numpy as np

from src.evaluation.benchmark import compare_pipelines


def test_compare_pipelines_deep_wins_when_clearly_better():
    rng = np.random.default_rng(0)
    classical = rng.normal(loc=0.70, scale=0.02, size=50)
    deep = classical + 0.10 + rng.normal(scale=0.01, size=50)

    result = compare_pipelines(deep, classical)

    assert result.deep_wins
    assert result.deep_mean > result.classical_mean
    assert result.p_value < 0.05


def test_compare_pipelines_no_win_when_scores_equivalent():
    rng = np.random.default_rng(0)
    classical = rng.normal(loc=0.70, scale=0.05, size=50)
    deep = rng.normal(loc=0.70, scale=0.05, size=50)

    result = compare_pipelines(deep, classical)
    assert not result.deep_wins
