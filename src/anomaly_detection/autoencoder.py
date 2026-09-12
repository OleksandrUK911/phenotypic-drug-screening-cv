from __future__ import annotations

import torch
from torch import nn


class EmbeddingAutoencoder(nn.Module):
    """Small MLP autoencoder trained ONLY on control-well embeddings.

    Trained to reconstruct "normal" (control) phenotype embeddings; a treated
    well whose embedding reconstructs poorly is, by construction, phenotypically
    unusual relative to the untreated baseline — an alternative to the
    Mahalanobis/kNN distance scorers in distance_scoring.py, cheaper to score
    at inference time and able to capture nonlinear structure.
    """

    def __init__(self, input_dim: int = 512, bottleneck_dim: int = 32) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, bottleneck_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


@torch.no_grad()
def reconstruction_error(model: EmbeddingAutoencoder, embeddings: torch.Tensor) -> torch.Tensor:
    model.eval()
    reconstructed = model(embeddings)
    return torch.mean((reconstructed - embeddings) ** 2, dim=1)
