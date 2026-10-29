"""Shared pytest fixtures (fake model, fake loader)."""
import pytest

import torch 
from tests.fakes import FakeLoader, TinyEncoder
from torch.utils.data import DataLoader, TensorDataset
@pytest.fixture 
def fake_loader():
    images = torch.randn(100, 3, 64, 64)
    labels = torch.randint(0, 200, (100, ))
    dataset = TensorDataset(images, labels)
    return DataLoader(dataset=dataset, batch_size=16, shuffle=False)
    
@pytest.fixture 
def fake_encoder():
    return TinyEncoder(feature_dim=64, in_channels=3)

@pytest.fixture 
def fake_batch():
    images = torch.randn(4, 3, 64, 64)
    labels = torch.randint(0, 200, (4, ))
    return images, labels