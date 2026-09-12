from __future__ import annotations

import numpy as np
from skimage.measure import regionprops

CLASSICAL_FEATURE_NAMES = [
    "area",
    "perimeter",
    "eccentricity",
    "solidity",
    "extent",
    "mean_intensity",
    "intensity_std",
]


def extract_classical_features(image: np.ndarray, mask: np.ndarray) -> dict[int, dict[str, float]]:
    """Hand-crafted per-cell shape/intensity features, in the spirit of CellProfiler's
    default feature set (AreaShape + Intensity modules) — the classical baseline
    against which the deep SSL pipeline is benchmarked (TODO.md section 8).

    `image` is a single 2D intensity channel (e.g. one Cell Painting stain);
    `mask` is the integer instance-segmentation label image from `src/segmentation/`.
    """
    features: dict[int, dict[str, float]] = {}
    for region in regionprops(mask, intensity_image=image):
        pixel_values = image[mask == region.label]
        features[region.label] = {
            "area": float(region.area),
            "perimeter": float(region.perimeter),
            "eccentricity": float(region.eccentricity),
            "solidity": float(region.solidity),
            "extent": float(region.extent),
            "mean_intensity": float(pixel_values.mean()),
            "intensity_std": float(pixel_values.std()),
        }
    return features


def features_to_matrix(features: dict[int, dict[str, float]]) -> tuple[np.ndarray, list[int]]:
    labels = sorted(features)
    matrix = np.array([[features[label][name] for name in CLASSICAL_FEATURE_NAMES] for label in labels])
    return matrix, labels
