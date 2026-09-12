from __future__ import annotations

import numpy as np


def paired_bootstrap_pvalue(
    scores_a: np.ndarray, scores_b: np.ndarray, n_bootstrap: int = 2000, seed: int = 0
) -> float:
    """One-sided paired bootstrap test for whether `scores_a` exceeds `scores_b`.

    Used to test "does the deep pipeline actually beat CellProfiler, or could
    the observed gap be explained by resampling noise?" — an unqualified
    single-number comparison (e.g. "82% vs 78% accuracy") is a common
    credibility gap in ML portfolios (TODO.md section 21). Returns the
    fraction of bootstrap resamples where the mean paired difference is <= 0
    (i.e. a small p-value means A reliably beats B).
    """
    if len(scores_a) != len(scores_b):
        raise ValueError("scores_a and scores_b must be paired (same length)")

    diffs = scores_a - scores_b
    rng = np.random.default_rng(seed)
    n = len(diffs)

    boot_means = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        sample = diffs[rng.integers(0, n, size=n)]
        boot_means[i] = sample.mean()

    return float(np.mean(boot_means <= 0.0))
