"""Encoder registry/factory: build_encoder(name, **kwargs)."""

from typing import Dict, Any
import torch.nn as nn
from .encoder import BaseEncoder
from .classifier import ImageClassifier


# Registry of encoder builders
_ENCODER_REGISTRY: Dict[str, callable] = {}


def register_encoder(name: str):
    """Decorator to register an encoder builder."""
    def decorator(fn):
        _ENCODER_REGISTRY[name] = fn
        return fn
    return decorator


def build_encoder(name: str, **kwargs) -> BaseEncoder:
    """
    Build an encoder by name.

    Args:
        name: Encoder name (e.g., "scratch_cnn", "resnet50", "efficientnet_b0").
        **kwargs: Arguments passed to the encoder constructor.

    Returns:
        BaseEncoder instance.
    """
    if name not in _ENCODER_REGISTRY:
        raise ValueError(f"Unknown encoder: {name}. Available: {list(_ENCODER_REGISTRY.keys())}")
    return _ENCODER_REGISTRY[name](**kwargs)


# Built-in encoders
@register_encoder("scratch_cnn")
def _build_scratch_cnn(feature_dim: int = 512, in_channels: int = 3) -> BaseEncoder:
    from .scratch_cnn import ScratchCNN
    return ScratchCNN(feature_dim=feature_dim, in_channels=in_channels)


@register_encoder("resnet50")
def _build_resnet50(feature_dim: int = 2048, pretrained: bool = True) -> BaseEncoder:
    from .resnet import ResNetEncoder
    return ResNetEncoder(feature_dim=feature_dim, pretrained=pretrained)


@register_encoder("efficientnet_b0")
def _build_efficientnet_b0(feature_dim: int = 1280, pretrained: bool = True) -> BaseEncoder:
    from .efficientnet import EfficientNetEncoder
    return EfficientNetEncoder(feature_dim=feature_dim, pretrained=pretrained)