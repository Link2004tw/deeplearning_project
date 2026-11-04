"""predict_all(model, loader, device) -> (y_true, y_pred, y_prob)."""

from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np


def _tta_logits(model: nn.Module, images: torch.Tensor, device: torch.device,
                tta_transforms: list) -> torch.Tensor:
    """Average logits over test-time augmentations."""
    logits_list = []
    with torch.no_grad():
        for tf in tta_transforms:
            x_aug = tf(images)
            logits = model(x_aug)
            logits_list.append(logits)
    return torch.stack(logits_list, dim=0).mean(dim=0)


def predict_all(model: nn.Module, loader: DataLoader, device: torch.device,
                return_logits: bool = False, tta: bool = False,
                tta_transforms: Optional[list] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    model.to(device)

    if tta and tta_transforms is None:
        tta_transforms = [
            lambda x: x,
            lambda x: torch.flip(x, dims=[3]),  # horizontal flip
        ]

    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if tta and tta_transforms:
                logits = _tta_logits(model, images, device, tta_transforms)
            else:
                logits = model(images)

            probs = F.softmax(logits, dim=1)
            preds = probs.argmax(dim=1)

            all_labels.append(labels.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
            all_probs.append(logits.cpu().numpy() if return_logits else probs.cpu().numpy())

    y_true = np.concatenate(all_labels, axis=0)
    y_pred = np.concatenate(all_preds, axis=0)
    y_prob = np.concatenate(all_probs, axis=0)

    return y_true, y_pred, y_prob
