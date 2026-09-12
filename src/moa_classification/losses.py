from __future__ import annotations

import torch
import torch.nn.functional as F


def label_smoothing_cross_entropy(logits: torch.Tensor, targets: torch.Tensor, smoothing: float = 0.1) -> torch.Tensor:
    """Standard label smoothing — softens one-hot targets so the model isn't
    pushed to output extreme confidence for annotations that are known to be
    imperfect (published MOA labels have documented annotation noise).
    """
    n_classes = logits.shape[-1]
    log_probs = F.log_softmax(logits, dim=-1)
    true_dist = torch.full_like(log_probs, smoothing / (n_classes - 1))
    true_dist.scatter_(1, targets.unsqueeze(1), 1.0 - smoothing)
    return -(true_dist * log_probs).sum(dim=-1).mean()


def generalized_cross_entropy(logits: torch.Tensor, targets: torch.Tensor, q: float = 0.7) -> torch.Tensor:
    """Generalized Cross Entropy (Zhang & Sabuncu, 2018) — a noise-robust loss
    that interpolates between MAE (q->1, robust but slow to converge) and
    standard cross-entropy (q->0). Bounds the loss contribution of any single
    example, so a handful of mislabeled compounds can't dominate the gradient
    the way they would under plain cross-entropy. See TODO.md section 5.
    """
    probs = F.softmax(logits, dim=-1)
    p_true = probs.gather(1, targets.unsqueeze(1)).squeeze(1).clamp_min(1e-12)
    return ((1.0 - p_true.pow(q)) / q).mean()
