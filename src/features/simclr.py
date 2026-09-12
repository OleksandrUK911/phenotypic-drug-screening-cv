from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import resnet18


class ProjectionHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 512, out_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SimCLRModel(nn.Module):
    """ResNet18 backbone (adapted for the 5-stain input) + SimCLR projection head.

    The backbone's output (pre-projection) is what gets used downstream as the
    phenotypic embedding — the projection head only exists to shape the
    contrastive loss and is discarded at inference time (standard SimCLR practice).
    """

    def __init__(self, n_channels: int = 5, embedding_dim: int = 512, projection_dim: int = 128) -> None:
        super().__init__()
        backbone = resnet18(weights=None)
        backbone.conv1 = nn.Conv2d(n_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.projection_head = ProjectionHead(embedding_dim, out_dim=projection_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embedding = self.backbone(x)
        projection = self.projection_head(embedding)
        return embedding, projection

    @torch.no_grad()
    def embed(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        return self.backbone(x)


def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.5) -> torch.Tensor:
    """Normalized temperature-scaled cross-entropy loss (SimCLR).

    z1, z2 are projections of two augmented views of the same batch of crops;
    each sample's positive pair is the other view of itself, negatives are
    every other sample (and its other view) in the batch.
    """
    batch_size = z1.shape[0]
    z = torch.cat([z1, z2], dim=0)
    z = F.normalize(z, dim=1)

    sim = z @ z.T / temperature
    sim.fill_diagonal_(float("-inf"))

    positive_idx = torch.cat([torch.arange(batch_size, 2 * batch_size), torch.arange(0, batch_size)]).to(z.device)

    return F.cross_entropy(sim, positive_idx)
