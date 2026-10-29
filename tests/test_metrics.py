import numpy as np
from src.evaluation.metrics import compute_metrics, top_k_accuracy

def test_compute_metrics_has_all_keys():
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 1, 2, 0, 1, 2])
    y_prob = np.eye(3)[y_pred]
    m = compute_metrics(y_true, y_pred, y_prob)
    expected = {"accuracy", "top5_accuracy", "macro_precision", "macro_recall", 
                "macro_f1", "weighted_precision", "weighted_recall", "weighted_f1"}
    assert set(m.keys()) == expected