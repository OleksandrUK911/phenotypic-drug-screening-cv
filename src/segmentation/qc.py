from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SegmentationQC:
    n_cells: int
    median_area: float
    area_cv: float
    flagged: bool
    reason: str | None


def cell_areas(mask: np.ndarray) -> np.ndarray:
    labels, counts = np.unique(mask, return_counts=True)
    return counts[labels != 0].astype(np.float64)


def qc_mask(
    mask: np.ndarray,
    min_cells: int = 5,
    max_cells: int = 2000,
    max_area_cv: float = 2.0,
) -> SegmentationQC:
    """Flag a segmented image as likely-bad without a human looking at it.

    Catches the two most common HCS segmentation failure modes: near-empty
    masks (out-of-focus field, failed stain) and pathologically variable cell
    sizes (over/under-segmentation), rather than silently feeding bad masks
    into downstream feature extraction.
    """
    areas = cell_areas(mask)
    n_cells = len(areas)

    if n_cells < min_cells:
        return SegmentationQC(n_cells, 0.0, 0.0, True, f"too few cells ({n_cells} < {min_cells})")
    if n_cells > max_cells:
        return SegmentationQC(n_cells, 0.0, 0.0, True, f"too many objects ({n_cells} > {max_cells}), likely noise")

    median_area = float(np.median(areas))
    area_cv = float(np.std(areas) / median_area) if median_area > 0 else float("inf")
    if area_cv > max_area_cv:
        return SegmentationQC(n_cells, median_area, area_cv, True, f"area CV too high ({area_cv:.2f} > {max_area_cv})")

    return SegmentationQC(n_cells, median_area, area_cv, False, None)
