from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.evaluation.significance import paired_bootstrap_pvalue
from src.utils.metrics import bootstrap_ci


@dataclass(frozen=True)
class BenchmarkResult:
    deep_mean: float
    deep_ci: tuple[float, float]
    classical_mean: float
    classical_ci: tuple[float, float]
    p_value: float
    deep_wins: bool


def compare_pipelines(deep_scores: np.ndarray, classical_scores: np.ndarray, alpha: float = 0.05) -> BenchmarkResult:
    """Combine bootstrap CIs and a paired significance test into one benchmark verdict.

    `deep_wins` only flips to True when the deep pipeline's mean exceeds the
    classical baseline's AND the paired bootstrap p-value clears `alpha` —
    guards against reporting a "the deep model wins" headline number that
    doesn't actually hold up statistically (TODO.md section 21).
    """
    deep_mean = float(np.mean(deep_scores))
    classical_mean = float(np.mean(classical_scores))
    p_value = paired_bootstrap_pvalue(deep_scores, classical_scores)

    return BenchmarkResult(
        deep_mean=deep_mean,
        deep_ci=bootstrap_ci(deep_scores),
        classical_mean=classical_mean,
        classical_ci=bootstrap_ci(classical_scores),
        p_value=p_value,
        deep_wins=(deep_mean > classical_mean) and (p_value < alpha),
    )
