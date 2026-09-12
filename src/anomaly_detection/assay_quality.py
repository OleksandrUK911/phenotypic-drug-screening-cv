from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AssayQuality:
    z_factor: float
    ssmd: float
    quality: str


def z_factor(positive_control: np.ndarray, negative_control: np.ndarray) -> float:
    """Z' factor (Zhang, Chung & Oldenburg, 1999) — standard HCS assay-quality metric.

    Z' = 1 - 3*(sigma_pos + sigma_neg) / |mu_pos - mu_neg|.
    Conventionally: >0.5 excellent assay, 0-0.5 marginal, <=0 unusable (signal
    doesn't separate from control noise) — reported per plate rather than
    trusting hit calls blindly, per TODO.md section 4.
    """
    mu_p, mu_n = positive_control.mean(), negative_control.mean()
    sigma_p, sigma_n = positive_control.std(ddof=1), negative_control.std(ddof=1)
    denom = abs(mu_p - mu_n)
    if denom == 0:
        return float("-inf")
    return 1.0 - (3.0 * (sigma_p + sigma_n)) / denom


def ssmd(positive_control: np.ndarray, negative_control: np.ndarray) -> float:
    """Strictly standardized mean difference — more robust than Z' for small sample sizes.

    SSMD = (mu_pos - mu_neg) / sqrt(sigma_pos^2 + sigma_neg^2).
    """
    mu_p, mu_n = positive_control.mean(), negative_control.mean()
    var_p, var_n = positive_control.var(ddof=1), negative_control.var(ddof=1)
    denom = np.sqrt(var_p + var_n)
    if denom == 0:
        return float("inf") if mu_p != mu_n else 0.0
    return (mu_p - mu_n) / denom


def _classify_quality(z: float) -> str:
    if z > 0.5:
        return "excellent"
    if z > 0.0:
        return "marginal"
    return "unusable"


def assess_plate_quality(positive_control: np.ndarray, negative_control: np.ndarray) -> AssayQuality:
    z = z_factor(positive_control, negative_control)
    s = ssmd(positive_control, negative_control)
    return AssayQuality(z_factor=z, ssmd=s, quality=_classify_quality(z))
