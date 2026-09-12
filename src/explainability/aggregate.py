from __future__ import annotations

import numpy as np


def aggregate_cams_by_class(cams: list[np.ndarray], labels: list[int]) -> dict[int, np.ndarray]:
    """Average Grad-CAM heatmaps per predicted/true MOA class.

    A single cell's CAM is noisy; averaging across many cells sharing a class
    reveals the *consistent* morphological region driving that MOA call
    (e.g. nuclear region for DNA-damaging agents vs. cytoskeleton for
    microtubule disruptors) rather than one cell's idiosyncratic activation.
    """
    if len(cams) != len(labels):
        raise ValueError("cams and labels must be the same length")

    grouped: dict[int, list[np.ndarray]] = {}
    for cam, label in zip(cams, labels):
        grouped.setdefault(label, []).append(cam)

    return {label: np.mean(np.stack(class_cams), axis=0) for label, class_cams in grouped.items()}
