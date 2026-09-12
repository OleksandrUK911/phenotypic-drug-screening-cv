from __future__ import annotations

from collections.abc import Callable

import numpy as np


def bootstrap_ci(
    values: np.ndarray,
    statistic: Callable[[np.ndarray], float] = np.mean,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap confidence interval for an arbitrary statistic.

    Used to report e.g. "accuracy = 0.82 [0.77, 0.86]" instead of a bare point
    estimate, per TODO.md section 21 — a single number invites the question
    "is that actually better than the baseline, or noise?".
    """
    rng = np.random.default_rng(seed)
    n = len(values)
    boot_stats = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        sample = values[rng.integers(0, n, size=n)]
        boot_stats[i] = statistic(sample)

    alpha = (1.0 - ci) / 2.0
    lower, upper = np.quantile(boot_stats, [alpha, 1.0 - alpha])
    return float(lower), float(upper)
