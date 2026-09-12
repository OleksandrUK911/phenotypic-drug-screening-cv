from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


@dataclass(frozen=True)
class HitCallingResult:
    threshold: float
    auc: float
    tpr_at_threshold: float
    fpr_at_threshold: float


def evaluate_hit_calling(scores: np.ndarray, is_active: np.ndarray) -> HitCallingResult:
    """Pick an anomaly-score threshold via Youden's J statistic and report ROC-AUC.

    `is_active` are known active/inactive labels (e.g. from a reference set of
    validated hits) used only for evaluation here, not for training the
    anomaly scorer itself, which is unsupervised.
    """
    if len(np.unique(is_active)) < 2:
        raise ValueError("Need both positive and negative examples to evaluate hit calling.")

    auc = roc_auc_score(is_active, scores)
    fpr, tpr, thresholds = roc_curve(is_active, scores)
    youden_j = tpr - fpr
    best_idx = int(np.argmax(youden_j))

    return HitCallingResult(
        threshold=float(thresholds[best_idx]),
        auc=float(auc),
        tpr_at_threshold=float(tpr[best_idx]),
        fpr_at_threshold=float(fpr[best_idx]),
    )


def call_hits(scores: np.ndarray, threshold: float) -> np.ndarray:
    return scores >= threshold
