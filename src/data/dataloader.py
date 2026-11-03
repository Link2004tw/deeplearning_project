"""Data loading: build_dataloaders for CUB-200."""

from typing import Dict, Any, Tuple, List
import torch
from torch.utils.data import DataLoader, Dataset


class CUBDataset(Dataset):
    """CUB-200-2011 dataset with images and captions."""

    def __init__(
        self,
        root: str,
        split: str = "train",
        transform=None,
        caption_transform=None,
    ):
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, idx: int):
        raise NotImplementedError


def get_transforms(cfg: Dict[str, Any], split: str = "train"):
    raise NotImplementedError


def build_dataloaders(cfg: Dict[str, Any]) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, Any]]:
    """
    Build train/val/test dataloaders.

    Returns:
        (train_loader, val_loader, test_loader, meta)
        meta contains: class_names, num_classes
    """
    raise NotImplementedError