from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi


@dataclass(frozen=True)
class CellCrop:
    label: int
    crop: np.ndarray
    centroid_yx: tuple[float, float]
    area: int


def extract_cell_crops(image: np.ndarray, mask: np.ndarray, crop_size: int = 96) -> list[CellCrop]:
    """Cut a fixed-size (C, crop_size, crop_size) patch centered on each cell.

    Fixed-size crops (rather than tight bounding boxes) keep downstream batch
    shapes uniform for the SSL/feature-extraction step, and give consistent
    context around each cell regardless of its own size.
    """
    half = crop_size // 2
    _, h, w = image.shape
    padded = np.pad(image, ((0, 0), (half, half), (half, half)), mode="reflect")

    crops = []
    labels = np.unique(mask)
    labels = labels[labels != 0]
    centroids = ndi.center_of_mass(mask > 0, mask, labels)
    areas = dict(zip(*np.unique(mask, return_counts=True)))

    for label, (cy, cx) in zip(labels, centroids):
        y0, x0 = int(round(cy)), int(round(cx))
        crop = padded[:, y0 : y0 + crop_size, x0 : x0 + crop_size]
        if crop.shape[1:] != (crop_size, crop_size):
            continue
        crops.append(CellCrop(label=int(label), crop=crop, centroid_yx=(cy, cx), area=int(areas[label])))
    return crops
