from typing import List, Tuple, Optional
import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from .gradcam import GradCAM, overlay_heatmap, denormalize_image


def select_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    n_correct: int = 4,
    n_wrong: int = 4,
) -> Tuple[np.ndarray, np.ndarray]:

    correct_mask = (y_true == y_pred)
    wrong_mask = ~correct_mask

    # Correct: highest confidence
    correct_conf = y_prob[correct_mask].max(axis=1)
    correct_idx = np.where(correct_mask)[0]
    correct_sorted = correct_idx[np.argsort(-correct_conf)]
    selected_correct = correct_sorted[:n_correct]

    # Wrong: highest confidence (most confident mistakes)
    wrong_conf = y_prob[wrong_mask].max(axis=1)
    wrong_idx = np.where(wrong_mask)[0]
    wrong_sorted = wrong_idx[np.argsort(-wrong_conf)]
    selected_wrong = wrong_sorted[:n_wrong]

    return selected_correct, selected_wrong


def generate_gradcam_grid(
        model: nn.Module,
        loader: DataLoader,
        device: torch.device,
        class_names: List[str],
        target_layer: nn.Module,
        n_correct: int = 4,
        n_wrong: int = 4,
        save_path: Optional[str] = None,
        denormalize_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        denormalize_std: Tuple[float, float, float] = (0.229, 0.224, 0.225), 

)-> plt.Figure:
    model.eval()
    all_images = []
    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for images, labels in loader: 
            images = images.to(device)
            logits = model(images)
            probs = torch.softmax(logits, dim=1)
            preds = probs.argmax(dim=1)


            all_images.append(images.cpu())
            all_labels.append(labels)
            all_preds.append(preds.cpu())
            all_probs.append(probs.cpu())

    images = torch.cat(all_images, dim=0)
    y_true = torch.cat(all_labels).numpy()
    y_pred = torch.cat(all_preds).numpy()
    y_prob = torch.cat(all_probs).numpy()

    correct_idx, wrong_idx = select_predictions(y_true, y_pred, y_prob, n_correct, n_wrong)
    selected_idx = np.concatenate([correct_idx, wrong_idx])
    selected_labels = ["correct"] * len(correct_idx) + ["wrong"] * len(wrong_idx)

    gradcam = GradCAM(model, target_layer)

    n_total = len(selected_idx)
    fig, axes = plt.subplots(n_total, 3, figsize=(12, 4 * n_total))
    if n_total == 1:
        axes = axes.reshape(1, 3)

    for row, idx in enumerate(selected_idx):
        img_tensor = images[idx]  # [C, H, W]
        true_label = y_true[idx]
        pred_label = y_pred[idx]
        confidence = y_prob[idx, pred_label]

        img_np = denormalize_image(img_tensor.numpy(), denormalize_mean, denormalize_std)

        heatmap = gradcam(img_tensor.unsqueeze(0), class_idx=pred_label)
        overlay = overlay_heatmap(img_np, heatmap)

        axes[row, 0].imshow(img_np)
        axes[row, 0].set_title(f"True: {class_names[true_label]}")
        axes[row, 0].axis("off")

        axes[row, 1].imshow(heatmap, cmap="jet", vmin=0, vmax=1)
        axes[row, 1].set_title(f"Pred: {class_names[pred_label]} ({confidence:.2f})")
        axes[row, 1].axis("off")

        axes[row, 2].imshow(overlay)
        axes[row, 2].set_title(f"{selected_labels[row].title()}")
        axes[row, 2].axis("off")

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")

    gradcam.remove_hooks()
    return fig 

def compare_models_gradcam(
        models: dict,
        loader: DataLoader,
        device: torch.device,
        class_names: List[str],
        n_samples: int = 4,
        save_path: Optional[str] = None
) -> plt.Figure:
    model_names = list(models.keys())
    n_models = len(model_names)
    first_model, first_layer = models[model_names[0]]
    first_model.eval()

    all_images = []
    all_labels = []
    with torch.no_grad():
        for images, labels in loader:
            all_images.append(images)
            all_labels.append(labels)
    images = torch.cat(all_images, dim=0)[:n_samples]
    labels = torch.cat(all_labels)[:n_samples]

    fig, axes = plt.subplots(n_samples, n_models + 1, figsize=(4 * (n_models + 1), 4 * n_samples))

    if n_samples == 1:
        axes = axes.reshape(1, -1)

    for row in range(n_samples):
        img_tensor = images[row].to(device=device)
        true_label = labels[row].item()

        img_np = denormalize_image(images[row].numpy())
        axes[row, 0].imshow(img_np)
        axes[row, 0].set_title(f"True: {class_names[true_label]}")
        axes[row, 0].axis("off")

        for col, model_name in enumerate(model_names):
            model, target_layer = models[model_name]
            model.eval()

            gradcam = GradCAM(model, target_layer)
            with torch.no_grad():
                logits = model(img_tensor.unsqueeze(0))
                pred = logits.argmax(dim=1).item()
                conf = torch.softmax(logits, dim=1)[0, pred].item()

            heatmap = gradcam(img_tensor.unsqueeze(0), class_idx=pred)
            overlay = overlay_heatmap(img_np, heatmap)

            axes[row, col + 1].imshow(overlay)
            axes[row, col + 1].set_title(f"{model_name}\nPred: {class_names[pred]} ({conf:.2f})")
            axes[row, col + 1].axis("off")

            gradcam.remove_hooks()

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=200, bbox_inches="tight")

        return fig 
