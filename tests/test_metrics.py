import numpy as np
import pytest
import pandas as pd

from src.evaluation.metrics import compute_metrics, top_k_accuracy, per_class_report, get_best_worst_classes


def test_compute_metrics_perfect():
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 1, 2, 0, 1, 2])
    y_prob = np.eye(3)[y_pred]

    metrics = compute_metrics(y_true, y_pred, y_prob)

    assert metrics["accuracy"] == 1.0
    assert metrics["top5_accuracy"] == 1.0
    assert metrics["macro_f1"] == 1.0
    assert metrics["weighted_f1"] == 1.0
    assert metrics["macro_precision"] == 1.0
    assert metrics["macro_recall"] == 1.0
    assert metrics["weighted_precision"] == 1.0
    assert metrics["weighted_recall"] == 1.0


def test_compute_metrics_known_confusion():
    y_true = np.array([0, 0, 0, 0, 1, 1, 2, 2, 2])
    y_pred = np.array([0, 0, 0, 1, 1, 1, 2, 0, 0])

    y_prob = np.zeros((9, 3))
    for i, p in enumerate(y_pred):
        y_prob[i, p] = 1.0

    metrics = compute_metrics(y_true, y_pred, y_prob)

    assert abs(metrics["accuracy"] - 6/9) < 1e-6
    assert abs(metrics["macro_f1"] - 0.656) < 0.02
    assert abs(metrics["weighted_f1"] - 0.656) < 0.02
    assert "macro_precision" in metrics
    assert "macro_recall" in metrics
    assert "weighted_precision" in metrics
    assert "weighted_recall" in metrics


def test_top_k_accuracy():
    y_true = np.array([0, 1, 2])
    y_prob = np.array([
        [0.7, 0.2, 0.1],  # true=0, top1=0
        [0.1, 0.7, 0.2],  # true=1, top1=1
        [0.2, 0.1, 0.7],  # true=2, top1=2
    ])

    assert top_k_accuracy(y_true, y_prob, k=1) == 1.0
    assert top_k_accuracy(y_true, y_prob, k=2) == 1.0
    assert top_k_accuracy(y_true, y_prob, k=3) == 1.0


def test_per_class_report():
    y_true = np.array([0, 0, 1, 1, 2])
    y_pred = np.array([0, 1, 1, 1, 2])
    class_names = ["A", "B", "C"]

    df = per_class_report(y_true, y_pred, class_names)

    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["class", "precision", "recall", "f1", "support"]
    assert len(df) == 3
    assert list(df["class"]) == ["A", "B", "C"]
    assert abs(df.loc[df["class"] == "A", "f1"].values[0] - 0.667) < 0.02
    assert abs(df.loc[df["class"] == "B", "f1"].values[0] - 0.8) < 0.02
    assert abs(df.loc[df["class"] == "C", "f1"].values[0] - 1.0) < 0.02


def test_get_best_worst_classes():
    df = pd.DataFrame({
        "class": ["A", "B", "C", "D"],
        "f1": [0.9, 0.5, 0.7, 0.3],
        "precision": [0.9, 0.5, 0.7, 0.3],
        "recall": [0.9, 0.5, 0.7, 0.3],
        "support": [10, 10, 10, 10],
    })

    best, worst = get_best_worst_classes(df, n=2, metric="f1")
    assert list(best["class"]) == ["A", "C"]
    assert set(worst["class"]) == {"B", "D"}


def test_zero_division_handling():
    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_pred = np.array([0, 0, 0, 0, 0, 0])
    y_prob = np.array([
        [0.9, 0.05, 0.05],
        [0.8, 0.1, 0.1],
        [0.7, 0.2, 0.1],
        [0.6, 0.3, 0.1],
        [0.5, 0.3, 0.2],
        [0.4, 0.3, 0.3],
    ])

    metrics = compute_metrics(y_true, y_pred, y_prob)
    assert "macro_f1" in metrics
    assert metrics["macro_f1"] >= 0.0
    assert metrics["weighted_f1"] >= 0.0