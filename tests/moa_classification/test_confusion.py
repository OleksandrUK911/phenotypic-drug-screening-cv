import numpy as np

from src.moa_classification.confusion import family_level_accuracy, per_class_accuracy


def test_per_class_accuracy_perfect_predictions():
    y_true = np.array([0, 0, 1, 1, 2])
    y_pred = np.array([0, 0, 1, 1, 2])
    class_names = ["a", "b", "c"]

    acc = per_class_accuracy(y_true, y_pred, class_names)
    assert acc == {"a": 1.0, "b": 1.0, "c": 1.0}


def test_family_level_accuracy_forgives_within_family_confusion():
    class_names = ["microtubule_a", "microtubule_b", "dna_damage"]
    moa_family = {"microtubule_a": "microtubule", "microtubule_b": "microtubule", "dna_damage": "dna_damage"}

    y_true = np.array([0, 0, 1, 1, 2])
    y_pred = np.array([1, 0, 0, 1, 2])

    raw_acc = per_class_accuracy(y_true, y_pred, class_names)
    family_acc = family_level_accuracy(y_true, y_pred, class_names, moa_family)

    assert family_acc["microtubule"] == 1.0
    assert family_acc["dna_damage"] == 1.0
    assert raw_acc["microtubule_a"] < 1.0
