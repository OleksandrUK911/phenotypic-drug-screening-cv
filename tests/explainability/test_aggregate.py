import numpy as np

from src.explainability.aggregate import aggregate_cams_by_class


def test_aggregate_cams_by_class_averages_within_class():
    cams = [np.ones((2, 2)) * 1.0, np.ones((2, 2)) * 3.0, np.ones((2, 2)) * 10.0]
    labels = [0, 0, 1]

    result = aggregate_cams_by_class(cams, labels)

    assert np.allclose(result[0], np.ones((2, 2)) * 2.0)
    assert np.allclose(result[1], np.ones((2, 2)) * 10.0)


def test_aggregate_cams_by_class_rejects_mismatched_lengths():
    try:
        aggregate_cams_by_class([np.zeros((2, 2))], [0, 1])
    except ValueError:
        return
    raise AssertionError("Expected ValueError for mismatched lengths")
