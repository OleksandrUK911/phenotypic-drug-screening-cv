from __future__ import annotations

from collections import defaultdict

import numpy as np
from sklearn.metrics import confusion_matrix


def per_class_accuracy(y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str]) -> dict[str, float]:
    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
    with np.errstate(divide="ignore", invalid="ignore"):
        acc = np.diag(cm) / cm.sum(axis=1)
    return {name: float(a) if not np.isnan(a) else float("nan") for name, a in zip(class_names, acc)}


def family_level_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    moa_family: dict[str, str],
) -> dict[str, float]:
    """Roll individual MOA classes up into broader mechanism families before scoring.

    Raw per-class accuracy can look worse than reality when confusion happens
    between two closely-related MOAs (e.g. two microtubule destabilizers) —
    reporting family-level accuracy alongside raw accuracy avoids overstating
    how "wrong" such confusions really are.
    """
    correct_by_family: dict[str, int] = defaultdict(int)
    total_by_family: dict[str, int] = defaultdict(int)

    for true_idx, pred_idx in zip(y_true, y_pred):
        true_family = moa_family[class_names[true_idx]]
        pred_family = moa_family[class_names[pred_idx]]
        total_by_family[true_family] += 1
        if true_family == pred_family:
            correct_by_family[true_family] += 1

    return {family: correct_by_family[family] / total for family, total in total_by_family.items()}
