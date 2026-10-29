"""Plots: confusion matrix, training curves, model comparison."""

from typing import Dict, List, Optional, Tuple

import matplotlib

import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd 
import seaborn as sns 
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(
  y_true: np.ndarray,
  y_pred:np.ndarray,
  class_names: List[str],
  top_k_classes: Optional[int] = None,
  normalize: bool = True,
  figsize: Tuple[int, int] = (12, 10),
  save_path: Optional[str] = None,      
) -> plt.Figure:
    cm = confusion_matrix(y_pred=y_pred, y_true=y_true)

    if normalize:
        cm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        cm = np.nan_to_num(cm) 
        fmt = ".2f"
        vmax = 1.0
    else:
        fmt = "d"
        vmax = None
    
    if top_k_classes is not None and top_k_classes < len(class_names): 
        confusion_per_class = cm.sum(axis=1) - np.diag(cm)
        top_indices = np.argsort(confusion_per_class)[-top_k_classes:]
        top_indices = np.sort(top_indices)

        cm = cm[top_indices][:, top_indices]
        class_names_subset = [class_names[i] for i in top_indices]

    else:
        class_names_subset = class_names

    fig, ax, = plt.subplots(figsize=figsize)

    sns.heatmap(
        cm, 
        annot=False,
        fmt=fmt,
        cmap="Blues",
        xticklabels=class_names_subset,
        yticklabels=class_names_subset,
        vmax=vmax,
        square=True,
        cbar_kws={
            "label": "Normalized Count" if normalize else "Count"
        },
        ax=ax
    )

    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix" + (" (Normalized)" if normalize else ""))
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")

    return fig

def plot_training_curves(
        history: Dict[str, List[float]],
        title: str ="Training curves",
        figsize: Tuple[int, int] = (12, 4),
        save_path: Optional[str] = None
)-> plt.Figure: 
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    epochs = range(1, len(history["train_loss"]) +1)

    # hena el loss
    axes[0].plot(epochs, history["train_loss"], label="Train Loss", marker="o")
    axes[0].plot(epochs, history["val_loss"], label="Val Loss", marker="s")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # We hena el Accuracy 
    axes[1].plot(epochs, history["train_acc"], label="Train Acc", marker="o")
    axes[1].plot(epochs, history["val_acc"], label="Val Acc", marker="s")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(title)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")

    return fig


def plot_model_comparison(
    results: Dict[str, Dict[str, float]],
    metrics: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None,
) -> plt.Figure:
    
    if metrics is None:
        metrics = ["accuracy", "top5_accuracy", "macro_f1", "weighted_f1"]

    model_names = list(results.keys())
    x = np.arange(len(model_names))
    width = 0.8 / len(metrics)
    fig, ax = plt.subplots(figsize=figsize)


    for i, metric in enumerate(metrics):
        values = [results[name].get(metric, 0) for name in model_names]
        offset = (i - len(metrics) / 2 + 0.5) * width
        ax.bar(x + offset, values, width, label=metric.replace("_", " ").title())

    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=45, ha="right")
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_ylim(0, 1.05)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")

    return fig   
