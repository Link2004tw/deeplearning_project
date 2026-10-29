"""Classification metrics: accuracy, top-k, precision/recall/F1, reports."""


from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd 

from sklearn.metrics import (
    accuracy_score, 
    precision_recall_fscore_support,
    confusion_matrix,
    top_k_accuracy_score
)

def top_k_accuracy(y_true: np.ndarray, y_prob: np.ndarray, k: int = 5) -> float:
    return top_k_accuracy_score(y_true, y_prob, k=k)

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    class_names: Optional[List[str]] = None,    
) -> Dict[str, float]:
    metrics = {

    }

    metrics["accuracy"] = float(accuracy_score(y_true, y_pred))

    if y_prob is not None:
        metrics["top5_accuracy"] = float(top_k_accuracy(y_true, y_prob, k=5))
    else:
        metrics["top5_accuracy"] = None

    macro_prec, macro_rec, macro_f1, _ = precision_recall_fscore_support(
    y_true, y_pred, average="macro", zero_division=0
)
    weighted_prec, weighted_rec, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    metrics["macro_precision"] = float(macro_prec)
    metrics["weighted_recall"] = float(weighted_rec)
    metrics["weighted_f1"] = float(weighted_f1)
    metrics["weighted_precision"] = float(weighted_prec)
    metrics["macro_recall"] = float(macro_rec)
    metrics["macro_f1"] = float(macro_f1)

    return metrics

def per_class_report(y_true: np.ndarray, y_pred: np.ndarray, class_names: List[str]) -> pd.DataFrame: 
    prec, rec, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )

    df = pd.DataFrame({
        "class": class_names,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "support": support
    })

    return df 

def get_best_worst_classes(
        per_class_df: pd.DataFrame,
        n: int = 10,
        metric: str = "f1"
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    sorted_df = per_class_df.sort_values(metric, ascending=False)
    best = sorted_df.head(n).copy()
    worst = sorted_df.tail(n).copy()

    return best, worst


def evaluate_model(
        model,
        loader,
        device,
        class_names: List[str]
) -> Dict:
    from .predict import predict_all 
    
    y_true, y_pred, y_prob = predict_all(model, loader, device)
    metrics = compute_metrics(y_true, y_pred, y_prob, class_names)
    per_class = per_class_report(y_true, y_pred, class_names)

    return {
        "metrics": metrics,
        "per_class": per_class,
        "y_true": y_true,
        "y_pred": y_pred,
        "y_prob": y_prob,
    }