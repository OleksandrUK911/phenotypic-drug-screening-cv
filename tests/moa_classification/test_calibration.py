import numpy as np

from src.moa_classification.calibration import expected_calibration_error, reliability_diagram


def test_ece_zero_for_perfectly_calibrated_predictions():
    rng = np.random.default_rng(0)
    confidences = rng.uniform(0.5, 1.0, size=2000)
    correct = rng.uniform(size=2000) < confidences

    ece = expected_calibration_error(confidences, correct, n_bins=10)
    assert ece < 0.05


def test_ece_high_for_overconfident_predictions():
    confidences = np.full(200, 0.99)
    correct = np.zeros(200, dtype=bool)
    correct[:20] = True

    ece = expected_calibration_error(confidences, correct, n_bins=10)
    assert ece > 0.7


def test_reliability_diagram_bins_have_correct_counts():
    confidences = np.array([0.05, 0.15, 0.55, 0.95])
    correct = np.array([True, False, True, True])

    bins = reliability_diagram(confidences, correct, n_bins=10)
    assert sum(b.count for b in bins) == 4
