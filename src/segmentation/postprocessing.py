from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.segmentation import watershed


def filter_small_objects(mask: np.ndarray, min_area: int) -> np.ndarray:
    """Drop labeled regions smaller than `min_area` pixels (segmentation noise/debris)."""
    out = mask.copy()
    for label, area in zip(*np.unique(mask, return_counts=True)):
        if label == 0:
            continue
        if area < min_area:
            out[out == label] = 0
    return relabel_sequential(out)


def relabel_sequential(mask: np.ndarray) -> np.ndarray:
    labels = np.unique(mask)
    labels = labels[labels != 0]
    out = np.zeros_like(mask)
    for new_label, old_label in enumerate(labels, start=1):
        out[mask == old_label] = new_label
    return out


def split_touching_cells(mask: np.ndarray, min_distance: int = 7) -> np.ndarray:
    """Watershed-split merged instances within each labeled blob.

    Cellpose/U-Net masks occasionally merge adjacent cells into one instance
    under crowding; this re-splits them using a distance-transform watershed,
    a standard fix rather than accepting undercounted cells (see TODO.md
    segmentation QC item).
    """
    out = np.zeros_like(mask)
    next_label = 1
    for label in np.unique(mask):
        if label == 0:
            continue
        blob = mask == label
        distance = ndi.distance_transform_edt(blob)
        coords = peak_local_max(distance, min_distance=min_distance, labels=blob)
        if len(coords) <= 1:
            out[blob] = next_label
            next_label += 1
            continue

        markers = np.zeros_like(distance, dtype=int)
        for i, (y, x) in enumerate(coords, start=1):
            markers[y, x] = i
        split = watershed(-distance, markers, mask=blob)
        for sub_label in np.unique(split):
            if sub_label == 0:
                continue
            out[split == sub_label] = next_label
            next_label += 1
    return out
