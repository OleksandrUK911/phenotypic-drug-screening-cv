import numpy as np

from src.anomaly_detection.distance_scoring import knn_distance_scores, mahalanobis_scores


def test_mahalanobis_score_higher_for_far_point():
    rng = np.random.default_rng(0)
    controls = rng.normal(loc=0.0, scale=1.0, size=(200, 4))
    near_point = np.array([[0.1, -0.1, 0.05, 0.0]])
    far_point = np.array([[20.0, 20.0, 20.0, 20.0]])

    near_score = mahalanobis_scores(near_point, controls)[0]
    far_score = mahalanobis_scores(far_point, controls)[0]

    assert far_score > near_score


def test_knn_distance_score_higher_for_far_point():
    rng = np.random.default_rng(0)
    controls = rng.normal(loc=0.0, scale=1.0, size=(50, 3))
    points = np.array([[0.0, 0.0, 0.0], [50.0, 50.0, 50.0]])

    scores = knn_distance_scores(points, controls, k=5)
    assert scores[1] > scores[0]


def test_knn_distance_handles_k_larger_than_control_set():
    controls = np.zeros((3, 2))
    points = np.array([[1.0, 1.0]])
    scores = knn_distance_scores(points, controls, k=10)
    assert scores.shape == (1,)
