from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score


def classical_baseline_cv_scores(
    features: np.ndarray, labels: np.ndarray, n_splits: int = 5, seed: int = 0
) -> np.ndarray:
    """Cross-validated accuracy of a RandomForest on classical (CellProfiler-style)
    features — the baseline the deep SSL + MOA-classifier pipeline is compared
    against in `benchmark.py`. RandomForest is used (rather than tuning an SVM)
    because it's the conventional strong default for small tabular feature sets
    in this literature.
    """
    clf = RandomForestClassifier(n_estimators=200, random_state=seed)
    return cross_val_score(clf, features, labels, cv=n_splits)
