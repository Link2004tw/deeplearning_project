"""El evaluation module"""

from .predict import predict_all 
from .metrics import (compute_metrics, top_k_accuracy, per_class_report, evaluate_model)

__all__ = [
    "predict_all",
    "compute_metrics",
    "top_k_accuracy",
    "per_class_report",
    "evaluate_model",
]