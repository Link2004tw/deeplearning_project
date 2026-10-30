"""Tracker interface for experiment tracking."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional


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
    """MLflow implementation."""

    def __init__(self, experiment_name: str, run_name: Optional[str] = None):
        raise NotImplementedError

    def log_params(self, params: Dict[str, Any]) -> None:
        raise NotImplementedError

    def log_metrics(self, metrics: Dict[str, float], step: int) -> None:
        raise NotImplementedError

    def log_artifact(self, path: str) -> None:
        raise NotImplementedError

    def finish(self) -> None:
        raise NotImplementedError


class WandbTracker(Tracker):
    """Weights & Biases implementation (optional)."""

    def __init__(self, project: str, run_name: Optional[str] = None, config: Optional[Dict] = None):
        raise NotImplementedError

    def log_params(self, params: Dict[str, Any]) -> None:
        raise NotImplementedError

    def log_metrics(self, metrics: Dict[str, float], step: int) -> None:
        raise NotImplementedError

    def log_artifact(self, path: str) -> None:
        raise NotImplementedError

    def finish(self) -> None:
        raise NotImplementedError


def get_tracker(backend: str, **kwargs) -> Tracker:
    """Factory function to get tracker by name."""
    raise NotImplementedError