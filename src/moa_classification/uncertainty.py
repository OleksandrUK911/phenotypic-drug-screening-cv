from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class UncertaintyEstimate:
    mean_probs: torch.Tensor
    predictive_entropy: torch.Tensor
    epistemic_uncertainty: torch.Tensor


def _enable_dropout(model: nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()


@torch.no_grad()
def mc_dropout_predict(model: nn.Module, x: torch.Tensor, n_samples: int = 20) -> UncertaintyEstimate:
    """Run `n_samples` stochastic forward passes with dropout active at inference.

    Predictive entropy of the averaged distribution captures total uncertainty;
    the spread across samples (epistemic_uncertainty, mean variance across
    classes) isolates the model-uncertainty component — a compound the model
    has barely seen should show high epistemic uncertainty even if the mean
    prediction looks confident. See TODO.md section 5.
    """
    model.eval()
    _enable_dropout(model)

    all_probs = torch.stack([F.softmax(model(x), dim=-1) for _ in range(n_samples)])
    mean_probs = all_probs.mean(dim=0)

    entropy = -(mean_probs * torch.log(mean_probs.clamp_min(1e-12))).sum(dim=-1)
    epistemic = all_probs.var(dim=0).mean(dim=-1)

    return UncertaintyEstimate(mean_probs=mean_probs, predictive_entropy=entropy, epistemic_uncertainty=epistemic)


@torch.no_grad()
def ensemble_predict(models: list[nn.Module], x: torch.Tensor) -> UncertaintyEstimate:
    """Deep Ensembles variant of the same idea: disagreement across independently
    trained models instead of dropout masks. More expensive to train, generally
    better-calibrated uncertainty than MC Dropout alone.
    """
    for m in models:
        m.eval()

    all_probs = torch.stack([F.softmax(m(x), dim=-1) for m in models])
    mean_probs = all_probs.mean(dim=0)
    entropy = -(mean_probs * torch.log(mean_probs.clamp_min(1e-12))).sum(dim=-1)
    epistemic = all_probs.var(dim=0).mean(dim=-1)

    return UncertaintyEstimate(mean_probs=mean_probs, predictive_entropy=entropy, epistemic_uncertainty=epistemic)
