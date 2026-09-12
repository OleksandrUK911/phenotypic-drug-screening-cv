from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression

from src.active_learning.sampling import diversity_sampling, uncertainty_sampling


def _entropy_scores(clf: LogisticRegression, x: np.ndarray) -> np.ndarray:
    probs = clf.predict_proba(x)
    return -(probs * np.log(probs.clip(min=1e-12))).sum(axis=1)


def simulate_active_learning(
    embeddings: np.ndarray,
    labels: np.ndarray,
    test_embeddings: np.ndarray,
    test_labels: np.ndarray,
    strategy: str = "uncertainty",
    n_initial: int = 10,
    batch_size: int = 10,
    n_rounds: int = 5,
    seed: int = 0,
) -> list[float]:
    """Simulate a compound-prioritization loop: start from a small labeled subset,
    repeatedly pick the next batch to "screen" via `strategy`, and track held-out
    accuracy after each round.

    This is the standard active-learning benchmark shape (learning curve of
    accuracy vs. #labeled examples), reframed as compound triage — the
    AI-driven-discovery narrative from TODO.md section 6: does prioritizing by
    model uncertainty reach a given accuracy with fewer screened compounds than
    screening in a random/fixed order?
    """
    if strategy not in {"uncertainty", "diversity", "random"}:
        raise ValueError(f"Unknown strategy: {strategy}")

    rng = np.random.default_rng(seed)
    n = len(embeddings)
    labeled_mask = np.zeros(n, dtype=bool)
    initial_idx = rng.choice(n, size=min(n_initial, n), replace=False)
    labeled_mask[initial_idx] = True

    accuracies: list[float] = []
    for _ in range(n_rounds + 1):
        clf = LogisticRegression(max_iter=1000)
        clf.fit(embeddings[labeled_mask], labels[labeled_mask])
        accuracies.append(float(clf.score(test_embeddings, test_labels)))

        if labeled_mask.all():
            break

        if strategy == "random":
            candidate_idx = np.where(~labeled_mask)[0]
            n_pick = min(batch_size, len(candidate_idx))
            picked = rng.choice(candidate_idx, size=n_pick, replace=False)
        elif strategy == "uncertainty":
            scores = _entropy_scores(clf, embeddings)
            picked = uncertainty_sampling(scores, labeled_mask, batch_size)
        else:
            picked = diversity_sampling(embeddings, labeled_mask, batch_size)

        labeled_mask[picked] = True

    return accuracies
