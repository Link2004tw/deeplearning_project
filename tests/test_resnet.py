import pytest
import torch
import torch.nn as nn

from src.models import build_encoder
from src.models.resnet import ResNetEncoder


def test_resnet18_forward_and_cam_target():
    encoder = ResNetEncoder(model_name="resnet18", pretrained=False)
    x = torch.randn(2, 3, 64, 64)

    features = encoder.forward_features(x)
    output = encoder(x)

    assert features.shape == (2, 512, 2, 2)
    assert output.shape == (2, 512)
    assert type(encoder.cam_target_layer()).__name__ == "BasicBlock"


def test_resnet18_projection_matches_feature_dim():
    encoder = ResNetEncoder(
        model_name="resnet18",
        feature_dim=128,
        pretrained=False,
    )
    output = encoder(torch.randn(2, 3, 64, 64))

    assert isinstance(encoder.projection, nn.Linear)
    assert output.shape == (2, 128)


def test_resnet18_full_finetune():
    encoder = ResNetEncoder(model_name="resnet18", pretrained=False, freeze="none")
    assert all(param.requires_grad for param in encoder.backbone.parameters())


def test_resnet18_frozen_backbone():
    encoder = ResNetEncoder(
        model_name="resnet18",
        pretrained=False,
        freeze="backbone",
    )
    assert all(not param.requires_grad for param in encoder.backbone.parameters())


def test_resnet18_frozen_backbone_backward_compatible_flag():
    encoder = ResNetEncoder(
        model_name="resnet18",
        pretrained=False,
        freeze_backbone=True,
    )
    assert all(not param.requires_grad for param in encoder.backbone.parameters())


def test_resnet18_partial_finetune_last_block_only():
    encoder = ResNetEncoder(
        model_name="resnet18",
        pretrained=False,
        freeze="partial",
        unfreeze_last_n=1,
    )

    assert all(param.requires_grad for param in encoder.backbone.layer4.parameters())
    assert all(not param.requires_grad for param in encoder.backbone.layer3.parameters())
    assert all(not param.requires_grad for param in encoder.backbone.layer2.parameters())


def test_resnet18_registry():
    encoder = build_encoder("resnet18", pretrained=False)
    assert isinstance(encoder, ResNetEncoder)
    assert encoder.output_dim == 512


def test_resnet_invalid_options():
    with pytest.raises(ValueError):
        ResNetEncoder(model_name="resnet999", pretrained=False)

    with pytest.raises(ValueError):
        ResNetEncoder(model_name="resnet18", pretrained=False, freeze="invalid")
