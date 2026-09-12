import numpy as np

from src.active_learning.sampling import diversity_sampling, uncertainty_sampling


def test_uncertainty_sampling_picks_highest_scores_among_unlabeled():
    scores = np.array([0.1, 0.9, 0.5, 0.95, 0.2])
    labeled_mask = np.array([False, True, False, False, False])  # index 1 already labeled despite high score

    picked = uncertainty_sampling(scores, labeled_mask, batch_size=2)

    assert picked == [3, 2]


def test_uncertainty_sampling_returns_empty_when_all_labeled():
    scores = np.array([0.1, 0.9])
    labeled_mask = np.array([True, True])
    assert uncertainty_sampling(scores, labeled_mask, batch_size=2) == []


def test_uncertainty_sampling_caps_at_available_candidates():
    scores = np.array([0.1, 0.9, 0.5])
    labeled_mask = np.array([False, False, True])
    picked = uncertainty_sampling(scores, labeled_mask, batch_size=10)
    assert set(picked) == {0, 1}


def test_diversity_sampling_picks_farthest_point_first():
    embeddings = np.array([[0.0], [1.0], [2.0], [100.0]])
    labeled_mask = np.array([True, False, False, False])  # point at 0.0 already labeled

    picked = diversity_sampling(embeddings, labeled_mask, batch_size=1)

    assert picked == [3]


def test_diversity_sampling_spreads_out_batch():
    embeddings = np.array([[0.0], [1.0], [2.0], [50.0], [100.0]])
    labeled_mask = np.array([True, False, False, False, False])

    picked = diversity_sampling(embeddings, labeled_mask, batch_size=2)

    assert 4 in picked  # farthest point (100.0) must be picked
    assert len(set(picked)) == 2
