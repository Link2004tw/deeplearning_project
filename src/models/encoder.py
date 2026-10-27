"""BaseEncoder: forward -> features, feature_dim, cam_target_layer()."""

import torch 
import torch.nn as nn 

class BaseEncoder(nn.Module):
    """Ana hakteb stubb kamelo el ba2i"""

    def __init__(self, feature_dim: int):
        super().__init__()
        self.feature_dim = feature_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def cam_target_layer(self) -> nn.Module:
        raise NotImplementedError