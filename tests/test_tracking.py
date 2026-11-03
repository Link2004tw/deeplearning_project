import pytest
import torch
from mlflow.tracking import MlflowClient
from torch.utils.data import DataLoader, TensorDataset

from src.models import ImageClassifier, ScratchCNN
from src.training.tracking import MLflowTracker, get_tracker, _flatten_params
from src.training.trainer import Trainer


def test_flatten_params():
    flat = _flatten_params(
        {
            "train": {"lr": 0.001, "optimizer": "adam"},
            "seed": 42,
            "subset": None,
            "sizes": [128, 224],
        }
    )

    assert flat["train.lr"] == 0.001
    assert flat["train.optimizer"] == "adam"
    assert flat["seed"] == 42
    assert flat["subset"] == "null"
    assert flat["sizes"] == "[128, 224]"


def test_mlflow_tracker_logs_run(tmp_path):
    tracking_uri = (tmp_path / "mlruns").as_uri()
    tracker = MLflowTracker(
        experiment_name="tracker-test",
        run_name="run-1",
        tracking_uri=tracking_uri,
    )
    run_id = tracker.run_id

    tracker.log_params({"train": {"lr": 0.001}, "seed": 42, "flag": True})
    tracker.log_metrics({"val_acc": 0.5, "val_f1": 0.4, "ignored": None}, step=2)

    artifact = tmp_path / "artifact.txt"
    artifact.write_text("hello")
    tracker.log_artifact(str(artifact))
    tracker.finish()

    client = MlflowClient(tracking_uri=tracking_uri)
    run = client.get_run(run_id)

    assert run.data.params["train.lr"] == "0.001"
    assert run.data.params["seed"] == "42"
    assert run.data.metrics["val_acc"] == 0.5
    assert run.data.metrics["val_f1"] == 0.4
    assert "ignored" not in run.data.metrics
    assert "artifact.txt" in [item.path for item in client.list_artifacts(run_id)]


def test_get_tracker_none():
    assert get_tracker("none") is None
    assert get_tracker("") is None


def test_get_tracker_mlflow(tmp_path):
    tracker = get_tracker(
        "mlflow",
        experiment_name="factory-test",
        tracking_uri=(tmp_path / "mlruns").as_uri(),
    )
    assert isinstance(tracker, MLflowTracker)
    tracker.finish()


def test_get_tracker_unknown():
    with pytest.raises(ValueError):
        get_tracker("unknown")


def test_trainer_logs_metrics_and_checkpoint_to_mlflow(tmp_path):
    tracking_uri = (tmp_path / "mlruns").as_uri()
    tracker = MLflowTracker(
        experiment_name="trainer-integration",
        run_name="run-1",
        tracking_uri=tracking_uri,
    )
    run_id = tracker.run_id

    images = torch.randn(8, 3, 32, 32)
    labels = torch.randint(0, 2, (8,))
    loader = DataLoader(TensorDataset(images, labels), batch_size=4)
    model = ImageClassifier(ScratchCNN(feature_dim=16), num_classes=2)
    cfg = {
        "epochs": 1,
        "optimizer": "adam",
        "lr": 1e-3,
        "scheduler": "none",
        "run_name": "run-1",
        "output_dir": str(tmp_path / "checkpoints"),
    }

    tracker.log_params({"train": cfg})
    trainer = Trainer(model, loader, loader, cfg, tracker=tracker)
    trainer.fit()
    tracker.finish()

    client = MlflowClient(tracking_uri=tracking_uri)
    run = client.get_run(run_id)

    assert run.data.params["train.lr"] == "0.001"
    assert "val_acc" in run.data.metrics
    assert "val_f1" in run.data.metrics
    assert "best.pt" in [item.path for item in client.list_artifacts(run_id)]
