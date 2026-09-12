import numpy as np
from sklearn.datasets import make_classification

from src.evaluation.baseline_classifier import classical_baseline_cv_scores


def test_classical_baseline_cv_scores_reasonable_on_separable_data():
    features, labels = make_classification(
        n_samples=200, n_features=7, n_informative=5, n_classes=3, n_clusters_per_class=1, random_state=0
    )

    scores = classical_baseline_cv_scores(features, labels, n_splits=5, seed=0)

    assert scores.shape == (5,)
    assert scores.mean() > 0.5
