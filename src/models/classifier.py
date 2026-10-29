"""ImageClassifier: encoder + linear head -> logits."""

import torch
import torch.nn as nn
from .encoder import BaseEncoder


class ImageClassifier(nn.Module):
    """Wrapper combining an encoder with a classification head."""

    def __init__(self, encoder: BaseEncoder, num_classes: int):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(encoder.feature_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        logits = self.head(features)
        return logits