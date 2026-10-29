"""Trainer: fit() -> history, saves best checkpoint."""

from typing import Dict, List, Any, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class Trainer:
    """Training loop with validation, checkpointing, and tracking."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        cfg: Dict[str, Any],
        tracker: Optional[Any] = None,
    ):
        raise NotImplementedError

    def fit(self) -> Dict[str, List[float]]:
        raise NotImplementedError

    def _train_epoch(self, epoch: int):
        raise NotImplementedError

    def _validate_epoch(self):
        raise NotImplementedError

    def _save_checkpoint(self, epoch: int, val_acc: float, is_best: bool = False):
        raise NotImplementedError