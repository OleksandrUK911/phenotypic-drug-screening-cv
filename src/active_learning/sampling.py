from __future__ import annotations

import numpy as np


def uncertainty_sampling(scores: np.ndarray, labeled_mask: np.ndarray, batch_size: int) -> list[int]:
    """Pick the `batch_size` unlabeled indices with the highest uncertainty score
    (e.g. predictive entropy from `moa_classification.uncertainty`).

    This is the core "which compound should we screen next" decision in an
    AI-driven discovery loop: prioritize compounds the model is least sure
    about, rather than screening in a fixed/random order. See TODO.md section 6.
    """
    candidate_idx = np.where(~labeled_mask)[0]
    if len(candidate_idx) == 0:
        return []
    ranked = candidate_idx[np.argsort(-scores[candidate_idx])]
    return ranked[:batch_size].tolist()


def diversity_sampling(embeddings: np.ndarray, labeled_mask: np.ndarray, batch_size: int) -> list[int]:
    """Greedy farthest-point sampling in embedding space among unlabeled points.

    An alternative to uncertainty sampling that doesn't need a trained
    classifier at all: picks compounds whose *phenotype* is most different
    from what's already been screened, maximizing coverage of the embedding
    space rather than chasing decision-boundary ambiguity.
    """
    labeled_idx = list(np.where(labeled_mask)[0])
    candidate_idx = list(np.where(~labeled_mask)[0])
    selected: list[int] = []

    if not labeled_idx and candidate_idx:
        selected.append(candidate_idx.pop(0))
        labeled_idx.append(selected[0])

    while len(selected) < batch_size and candidate_idx:
        reference = embeddings[labeled_idx + selected]
        dists = np.linalg.norm(embeddings[candidate_idx][:, None, :] - reference[None, :, :], axis=-1)
        min_dists = dists.min(axis=1)
        best_local_idx = int(np.argmax(min_dists))
        selected.append(candidate_idx.pop(best_local_idx))

    return selected
