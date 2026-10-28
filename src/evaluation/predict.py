"""predict_all(model, loader, device) -> (y_true, y_pred, y_prob)."""

from typing import Tuple, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader 
import numpy as np 

def predict_all(model: nn.Module, loader: DataLoader, device: torch.device, return_logits: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    model.to(device)

    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = model(images)
            probs = torch.softmax(logits, dim=1)
            preds = probs.argmax(dim=1)

            all_labels.append(labels.cpu().numpy)
            all_preds.append(preds.cpu().numpy)
            all_probs.append(logits.cpu().numpy if return_logits else probs.cpu().numpy)

    y_true = np.concatenate(all_labels, axis=0)
    y_pred = np.concatenate(all_preds, axis=0)
    y_prob = np.concatenate(all_probs, axis=0)

    return y_true, y_pred, y_prob
                                 