"""Tracker interface and MLflow implementation."""

import math
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


def _flatten_params(params: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    flat: Dict[str, Any] = {}
    for key, value in params.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            flat.update(_flatten_params(value, name))
        elif isinstance(value, (list, tuple)):
            flat[name] = str(value)
        elif value is None:
            flat[name] = "null"
        else:
            flat[name] = value
    return flat


class Tracker(ABC):
    """Base tracker interface."""

    @abstractmethod
    def log_params(self, params: Dict[str, Any]) -> None:
        pass

    @abstractmethod
    def log_metrics(self, metrics: Dict[str, float], step: int) -> None:
        pass

    @abstractmethod
    def log_artifact(self, path: str) -> None:
        pass

    @abstractmethod
    def finish(self) -> None:
        pass


class MLflowTracker(Tracker):
    """MLflow implementation with lazy import and nested-parameter flattening."""

    def __init__(
        self,
        experiment_name: str,
        run_name: Optional[str] = None,
        tracking_uri: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
    ):
        import os

        os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

        import mlflow

        self._mlflow = mlflow
        if tracking_uri is not None:
            mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        if self._mlflow.active_run() is not None:
            self._mlflow.end_run()
        self._run = mlflow.start_run(run_name=run_name, tags=tags)
        self.run_id = self._run.info.run_id

    def log_params(self, params: Dict[str, Any]) -> None:
        self._mlflow.log_params(_flatten_params(params))

    def log_metrics(self, metrics: Dict[str, float], step: int) -> None:
        clean = {}
        for key, value in metrics.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            number = float(value)
            if math.isfinite(number):
                clean[str(key)] = number
        if clean:
            self._mlflow.log_metrics(clean, step=int(step))

    def log_artifact(self, path: str) -> None:
        self._mlflow.log_artifact(path)

    def finish(self) -> None:
        if self._mlflow.active_run() is not None:
            self._mlflow.end_run()


class WandbTracker(Tracker):
    """Weights & Biases implementation placeholder."""

    def __init__(
        self,
        project: str,
        run_name: Optional[str] = None,
        config: Optional[Dict] = None,
    ):
        raise NotImplementedError("W&B tracking is not implemented; use MLflow.")

    def log_params(self, params: Dict[str, Any]) -> None:
        raise NotImplementedError

    def log_metrics(self, metrics: Dict[str, float], step: int) -> None:
        raise NotImplementedError

    def log_artifact(self, path: str) -> None:
        raise NotImplementedError

    def finish(self) -> None:
        raise NotImplementedError


def get_tracker(backend: str, **kwargs) -> Optional[Tracker]:
    """Factory function to get a tracker by name."""
    name = str(backend).lower()
    if name in {"", "none", "null"}:
        return None
    if name == "mlflow":
        return MLflowTracker(**kwargs)
    if name == "wandb":
        return WandbTracker(**kwargs)
    raise ValueError(f"Unknown tracker backend: {backend}")
