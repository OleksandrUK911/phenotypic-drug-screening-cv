from __future__ import annotations

import numpy as np


def mahalanobis_scores(embeddings: np.ndarray, control_embeddings: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Mahalanobis distance of each embedding from the control-well distribution.

    Uses the control set's own covariance so the "unit" of distance accounts
    for which directions in embedding space are noisy-but-normal for controls
    vs. genuinely unusual — a plain Euclidean distance would treat both the same.
    """
    mean = control_embeddings.mean(axis=0)
    cov = np.cov(control_embeddings, rowvar=False) + eps * np.eye(control_embeddings.shape[1])
    inv_cov = np.linalg.inv(cov)

    diff = embeddings - mean
    return np.sqrt(np.einsum("ij,jk,ik->i", diff, inv_cov, diff))


def knn_distance_scores(embeddings: np.ndarray, control_embeddings: np.ndarray, k: int = 5) -> np.ndarray:
    """Mean distance to the k nearest control-well embeddings — a nonparametric
    alternative to Mahalanobis distance that doesn't assume a unimodal control
    distribution (useful if controls span more than one healthy phenotype cluster).
    """
    k = min(k, len(control_embeddings))
    dists = np.linalg.norm(embeddings[:, None, :] - control_embeddings[None, :, :], axis=-1)
    nearest = np.sort(dists, axis=1)[:, :k]
    return nearest.mean(axis=1)
