"""ResNet encoder wrapper for transfer learning."""

import torch
import torch.nn as nn
from .encoder import BaseEncoder


class ResNetEncoder(BaseEncoder):
    """ResNet encoder with configurable feature dimension."""

    def __init__(self, feature_dim: int = 2048, pretrained: bool = True, model_name: str = "resnet50"):
        super().__init__(feature_dim=feature_dim)
        self.feature_dim = feature_dim
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def cam_target_layer(self) -> nn.Module:
        raise NotImplementedError