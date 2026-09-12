from __future__ import annotations

import numpy as np

from src.data.schema import WellMetadata


def _whitening_transform(control_embeddings: np.ndarray, eps: float = 1e-6) -> tuple[np.ndarray, np.ndarray]:
    mean = control_embeddings.mean(axis=0)
    centered = control_embeddings - mean
    cov = np.cov(centered, rowvar=False) + eps * np.eye(centered.shape[1])
    u, s, _ = np.linalg.svd(cov)
    whitening = u @ np.diag(1.0 / np.sqrt(s)) @ u.T
    return mean, whitening


def typical_variation_normalization(
    well_embeddings: dict[tuple[str, str], np.ndarray],
    metadata: dict[tuple[str, str], WellMetadata],
) -> dict[tuple[str, str], np.ndarray]:
    """Per-plate whitening against that plate's own control (DMSO) wells (Ando et al. 2017 TVN).

    Removes plate-to-plate technical variation (illumination day, reagent lot,
    imaging session) by re-centering/whitening every well on a plate using the
    covariance structure of ONLY that plate's controls — so a "typical"
    (control-like) well maps to the same distribution on every plate,
    while genuine phenotypic shifts (treatment effects) survive the transform.
    See TODO.md section 3.
    """
    by_plate: dict[str, list[tuple[str, np.ndarray]]] = {}
    for (plate, well), emb in well_embeddings.items():
        by_plate.setdefault(plate, []).append((well, emb))

    corrected: dict[tuple[str, str], np.ndarray] = {}
    for plate, wells in by_plate.items():
        control_vecs = np.stack([emb for well, emb in wells if metadata[(plate, well)].is_control])
        if len(control_vecs) < 2:
            raise ValueError(f"Plate {plate} has fewer than 2 control wells; cannot estimate covariance.")

        mean, whitening = _whitening_transform(control_vecs)
        for well, emb in wells:
            corrected[(plate, well)] = (emb - mean) @ whitening

    return corrected
