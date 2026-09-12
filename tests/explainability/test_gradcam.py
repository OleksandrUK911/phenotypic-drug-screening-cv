import torch
from torch import nn

from src.explainability.gradcam import GradCAM


class _TinyCNN(nn.Module):
    def __init__(self, n_classes: int = 3) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(2, 4, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(4, 8, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(8, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.pool(x).flatten(1)
        return self.fc(x)


def test_gradcam_output_shape_and_range():
    torch.manual_seed(0)
    model = _TinyCNN()
    cam_tool = GradCAM(model, target_layer=model.conv2)

    x = torch.randn(1, 2, 16, 16, requires_grad=True)
    heatmap = cam_tool.generate(x, class_idx=0)

    assert heatmap.shape == (16, 16)
    assert heatmap.min() >= 0.0
    assert heatmap.max() <= 1.0 + 1e-6
