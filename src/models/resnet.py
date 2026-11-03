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
        unfreeze_blocks: int = 0,
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

        self.freeze = str(freeze).lower()
        self.unfreeze_last_n = int(unfreeze_last_n)
        self.unfreeze_blocks = int(unfreeze_blocks)
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

        # Finer control: unfreeze the last N blocks of the final stage (layer4).
        # Useful for fine-tuning just the top of the backbone while keeping the
        # rest frozen. Works on top of any freeze mode (default 0 = no change).
        if self.unfreeze_blocks > 0:
            layer4 = self.backbone.layer4
            n = max(0, min(self.unfreeze_blocks, len(layer4)))
            for block in layer4[len(layer4) - n :] if n else []:
                for param in block.parameters():
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
        """Return the last convolutional layer for Grad-CAM.

        ResNet's final residual block (layer4[-1]) is not itself a conv layer.
        Bottleneck blocks (resnet50/101/152) end with conv3; BasicBlock blocks
        (resnet18/34) end with conv2. Returning the actual conv layer keeps the
        Grad-CAM hooks on feature maps from the last conv, as required.
        """
        last_block = self.backbone.layer4[-1]
        if hasattr(last_block, "conv3"):
            return last_block.conv3
        return last_block.conv2
