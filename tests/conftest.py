"""Shared pytest fixtures (fake model, fake loader)."""
import pytest

import torch 
from tests.fakes import FakeLoader, TinyEncoder

@pytest.fixture 
def fake_loader():
    return FakeLoader(n_samples=512, batch_size=32, n_classes=200, image_size=64, seed=42)

@pytest.fixture 
def fake_encoder():
    return TinyEncoder(feature_dim=64, in_channels=3)

@pytest.fixture 
def fake_batch():
    images = torch.randn(8, 3, 64, 64)
    labels = torch.randint(0, 200, (8, ))
    return images, labels