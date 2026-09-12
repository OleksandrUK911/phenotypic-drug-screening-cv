from __future__ import annotations

import numpy as np


def cam_mass_inside_mask(cam: np.ndarray, cell_mask: np.ndarray) -> float:
    """Fraction of a Grad-CAM heatmap's total activation that falls inside the
    segmented cell region.

    A trustworthy explanation should mostly light up the cell body/nucleus,
    not plate-edge artifacts or background debris — this is a cheap automated
    proxy for that, flagged low if the CAM is background-heavy (TODO.md
    section 7 sanity check).
    """
    total = cam.sum()
    if total == 0:
        return 0.0
    return float(cam[cell_mask.astype(bool)].sum() / total)


def flag_background_heavy_explanations(
    cams: list[np.ndarray], cell_masks: list[np.ndarray], min_inside_fraction: float = 0.5
) -> list[int]:
    """Return indices of explanations whose activation mass is mostly outside the cell —
    likely unreliable and worth excluding from qualitative "here's what drives this MOA" figures.
    """
    flagged = []
    for i, (cam, mask) in enumerate(zip(cams, cell_masks)):
        if cam_mass_inside_mask(cam, mask) < min_inside_fraction:
            flagged.append(i)
    return flagged
