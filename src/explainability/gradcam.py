from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class GradCAM:
    """Grad-CAM (Selvaraju et al., 2017) for a CNN backbone's last conv layer.

    Built directly on backbone forward/backward hooks rather than the
    `grad-cam` package's default target layers, since the SimCLR backbone here
    is a ResNet18 stem adapted to 5-channel Cell Painting input — using the
    standard ResNet block naming keeps this compatible with that library if
    swapped in later, while staying dependency-light for now.
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model
        self._activations: torch.Tensor | None = None
        self._gradients: torch.Tensor | None = None

        target_layer.register_forward_hook(self._save_activations)
        target_layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, module: nn.Module, inp: tuple, out: torch.Tensor) -> None:
        self._activations = out.detach()

    def _save_gradients(self, module: nn.Module, grad_in: tuple, grad_out: tuple) -> None:
        self._gradients = grad_out[0].detach()

    def generate(self, x: torch.Tensor, class_idx: int) -> np.ndarray:
        """Return a (H, W) heatmap in [0, 1] for one input image and target class index."""
        self.model.zero_grad()
        logits = self.model(x)
        score = logits[:, class_idx].sum()
        score.backward()

        weights = self._gradients.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * self._activations).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)

        cam = cam.squeeze().cpu().numpy()
        cam -= cam.min()
        max_val = cam.max()
        return cam / max_val if max_val > 0 else cam
