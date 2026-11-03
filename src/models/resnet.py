"""ResNet encoder wrapper for transfer learning."""

from typing import Optional

import torch
import torch.nn as nn
from torchvision import models

from .encoder import BaseEncoder


_RESNET_BUILDERS = {
    "resnet18": (models.resnet18, models.ResNet18_Weights),
    "resnet34": (models.resnet34, models.ResNet34_Weights),
    "resnet50": (models.resnet50, models.ResNet50_Weights),
    "resnet101": (models.resnet101, models.ResNet101_Weights),
    "resnet152": (models.resnet152, models.ResNet152_Weights),
}


class ResNetEncoder(BaseEncoder):
    """ResNet encoder with optional projection and fine-tuning modes."""

    def __init__(
        self,
        model_name: str = "resnet18",
        feature_dim: Optional[int] = None,
        pretrained: bool = True,
        freeze: str = "none",
        unfreeze_last_n: int = 1,
        freeze_backbone: Optional[bool] = None,
    ):
        if model_name not in _RESNET_BUILDERS:
            raise ValueError(
                f"Unknown ResNet model: {model_name}. "
                f"Available: {list(_RESNET_BUILDERS.keys())}"
            )

        builder, weights_enum = _RESNET_BUILDERS[model_name]
        backbone = builder(weights=weights_enum.DEFAULT if pretrained else None)
        backbone_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()

        resolved_dim = int(feature_dim) if feature_dim is not None else backbone_dim
        super().__init__(feature_dim=resolved_dim)

        self.model_name = model_name
        self.backbone = backbone
        self.backbone_dim = backbone_dim
        self.projection = (
            nn.Identity()
            if resolved_dim == backbone_dim
            else nn.Linear(backbone_dim, resolved_dim)
        )

        if freeze_backbone is not None:
            freeze = "backbone" if freeze_backbone else "none"
        self.freeze = str(freeze).lower()
        self.unfreeze_last_n = int(unfreeze_last_n)
        self._apply_freeze()

    def _apply_freeze(self) -> None:
        if self.freeze in {"none", "full", "false"}:
            return
        if self.freeze not in {"backbone", "partial"}:
            raise ValueError(
                f"Unknown freeze mode: {self.freeze}. "
                "Use 'none', 'backbone', or 'partial'."
            )

        for param in self.backbone.parameters():
            param.requires_grad = False

        if self.freeze == "partial":
            stages = [
                self.backbone.layer1,
                self.backbone.layer2,
                self.backbone.layer3,
                self.backbone.layer4,
            ]
            count = max(0, min(self.unfreeze_last_n, len(stages)))
            for stage in stages[len(stages) - count :] if count else []:
                for param in stage.parameters():
                    param.requires_grad = True

        for param in self.projection.parameters():
            param.requires_grad = True

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        return self.projection(x)

    def cam_target_layer(self) -> nn.Module:
        return self.backbone.layer4[-1]
