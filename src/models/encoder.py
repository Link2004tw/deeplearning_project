"""BaseEncoder stub for classification."""

import torch
import torch.nn as nn

class BaseEncoder(nn.Module):
    """Base encoder interface: forward -> features, feature_dim, cam_target_layer()."""

    def __init__(self, feature_dim: int):
        super().__init__()
        self.feature_dim = feature_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def cam_target_layer(self) -> nn.Module:
        raise NotImplementedError

    @property
    def output_dim(self) -> int:
        return self.feature_dim