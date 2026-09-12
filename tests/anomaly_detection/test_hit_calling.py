import numpy as np

from src.anomaly_detection.hit_calling import call_hits, evaluate_hit_calling


def test_evaluate_hit_calling_perfect_separation():
    scores = np.array([0.1, 0.2, 0.9, 0.95, 0.3, 0.85])
    is_active = np.array([0, 0, 1, 1, 0, 1])

    result = evaluate_hit_calling(scores, is_active)

    assert result.auc == 1.0
    assert result.tpr_at_threshold == 1.0
    assert result.fpr_at_threshold == 0.0


def test_evaluate_hit_calling_requires_both_classes():
    scores = np.array([0.1, 0.2, 0.3])
    is_active = np.array([0, 0, 0])

    try:
        evaluate_hit_calling(scores, is_active)
    except ValueError:
        return
    raise AssertionError("Expected ValueError when only one class present")


def test_call_hits_thresholding():
    scores = np.array([0.1, 0.5, 0.9])
    hits = call_hits(scores, threshold=0.5)
    assert list(hits) == [False, True, True]
