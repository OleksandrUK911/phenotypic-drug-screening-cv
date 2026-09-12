from __future__ import annotations

import torch
from torch import nn


class MOAClassifier(nn.Module):
    """MLP classifier over batch-corrected well-level phenotypic embeddings.

    Dropout is kept active at inference time (see uncertainty.py) to enable
    MC Dropout uncertainty estimates without needing a separate ensemble.
    """

    def __init__(self, input_dim: int = 512, hidden_dim: int = 256, n_classes: int = 12, dropout: float = 0.3) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
