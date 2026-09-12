import torch

from src.moa_classification.losses import generalized_cross_entropy, label_smoothing_cross_entropy


def test_label_smoothing_loss_positive_and_finite():
    logits = torch.randn(8, 5)
    targets = torch.randint(0, 5, (8,))

    loss = label_smoothing_cross_entropy(logits, targets, smoothing=0.1)
    assert loss.item() > 0
    assert torch.isfinite(loss)


def test_label_smoothing_exceeds_plain_cross_entropy_for_confident_correct_logits():
    logits = torch.zeros(4, 3)
    logits[:, 0] = 10.0
    targets = torch.zeros(4, dtype=torch.long)

    smoothed = label_smoothing_cross_entropy(logits, targets, smoothing=0.1)
    plain = torch.nn.functional.cross_entropy(logits, targets)
    assert smoothed.item() > plain.item()


def test_generalized_cross_entropy_bounded_by_one_over_q():
    logits = torch.randn(16, 4)
    targets = torch.randint(0, 4, (16,))
    q = 0.7

    loss = generalized_cross_entropy(logits, targets, q=q)
    assert 0.0 <= loss.item() <= 1.0 / q


def test_generalized_cross_entropy_near_zero_for_confident_correct_prediction():
    logits = torch.zeros(1, 3)
    logits[0, 1] = 20.0
    targets = torch.tensor([1])

    loss = generalized_cross_entropy(logits, targets, q=0.7)
    assert loss.item() < 1e-3
