from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ReliabilityBin:
    confidence: float
    accuracy: float
    count: int


def reliability_diagram(confidences: np.ndarray, correct: np.ndarray, n_bins: int = 10) -> list[ReliabilityBin]:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    results = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (confidences > lo) & (confidences <= hi)
        if not mask.any():
            continue
        results.append(
            ReliabilityBin(
                confidence=float(confidences[mask].mean()),
                accuracy=float(correct[mask].mean()),
                count=int(mask.sum()),
            )
        )
    return results


def expected_calibration_error(confidences: np.ndarray, correct: np.ndarray, n_bins: int = 10) -> float:
    """ECE: confidence-weighted average gap between predicted confidence and empirical accuracy.

    A well-calibrated MOA classifier is important for pharma decision-making —
    "70% confident" predictions should be right about 70% of the time, not just
    for the model's overall accuracy to look good. See TODO.md section 5.
    """
    n = len(confidences)
    ece = 0.0
    for b in reliability_diagram(confidences, correct, n_bins):
        ece += (b.count / n) * abs(b.confidence - b.accuracy)
    return ece
