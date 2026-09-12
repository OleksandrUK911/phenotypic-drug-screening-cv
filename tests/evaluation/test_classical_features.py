import numpy as np

from src.evaluation.classical_features import CLASSICAL_FEATURE_NAMES, extract_classical_features, features_to_matrix


def test_extract_classical_features_area_matches_known_square():
    mask = np.zeros((20, 20), dtype=int)
    mask[2:12, 2:12] = 1  # 10x10 square = 100 px
    image = np.ones((20, 20), dtype=np.float64) * 5.0

    features = extract_classical_features(image, mask)

    assert features[1]["area"] == 100.0
    assert features[1]["mean_intensity"] == 5.0
    assert features[1]["intensity_std"] == 0.0


def test_extract_classical_features_handles_multiple_labels():
    mask = np.zeros((20, 20), dtype=int)
    mask[0:5, 0:5] = 1
    mask[10:15, 10:15] = 2
    image = np.random.default_rng(0).random((20, 20))

    features = extract_classical_features(image, mask)
    assert set(features.keys()) == {1, 2}


def test_features_to_matrix_shape_and_column_order():
    mask = np.zeros((10, 10), dtype=int)
    mask[0:4, 0:4] = 1
    image = np.ones((10, 10))

    features = extract_classical_features(image, mask)
    matrix, labels = features_to_matrix(features)

    assert matrix.shape == (1, len(CLASSICAL_FEATURE_NAMES))
    assert labels == [1]
