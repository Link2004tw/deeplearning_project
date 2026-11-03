import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.models import ImageClassifier, ScratchCNN
from src.training.trainer import Trainer


def make_loader(num_samples=16, num_classes=2, image_size=32, batch_size=4):
    images = torch.randn(num_samples, 3, image_size, image_size)
    labels = torch.randint(0, num_classes, (num_samples,))
    return DataLoader(TensorDataset(images, labels), batch_size=batch_size)


class ConstantModel(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(num_classes))

    def forward(self, x):
        return self.bias.expand(x.size(0), -1)


def test_trainer_history_and_checkpoints(tmp_path):
    model = ImageClassifier(ScratchCNN(feature_dim=32), num_classes=2)
    loader = make_loader()
    cfg = {
        "epochs": 2,
        "optimizer": "adam",
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "scheduler": "cosine",
        "amp": True,
        "run_name": "trainer_test",
        "output_dir": str(tmp_path / "checkpoints"),
        "early_stopping": {"monitor": "val_acc", "patience": 5},
    }

    trainer = Trainer(
        model,
        loader,
        loader,
        cfg,
        tracker=None,
        metadata={"class_names": ["a", "b"], "image_size": 32, "encoder": "scratch_cnn"},
    )
    history = trainer.fit()

    expected_keys = {"train_loss", "train_acc", "val_loss", "val_acc", "val_f1", "lr"}
    assert expected_keys.issubset(history.keys())
    assert all(len(values) == 2 for values in history.values())
    assert history["lr"][-1] < cfg["lr"]

    checkpoint_dir = tmp_path / "checkpoints"
    assert (checkpoint_dir / "best.pt").exists()
    assert (checkpoint_dir / "last.pt").exists()

    checkpoint = torch.load(checkpoint_dir / "best.pt", map_location="cpu")
    assert checkpoint["metadata"]["class_names"] == ["a", "b"]
    assert checkpoint["config"]["run_name"] == "trainer_test"
    assert "model_state_dict" in checkpoint
    assert "optimizer_state_dict" in checkpoint


def test_trainer_early_stopping(tmp_path):
    model = ConstantModel(num_classes=2)
    loader = make_loader()
    cfg = {
        "epochs": 5,
        "optimizer": "adam",
        "lr": 0.0,
        "scheduler": "none",
        "run_name": "early_stop",
        "output_dir": str(tmp_path / "checkpoints"),
        "early_stopping": {"monitor": "val_acc", "patience": 1},
    }

    trainer = Trainer(model, loader, loader, cfg)
    history = trainer.fit()

    assert len(history["train_loss"]) == 2
    assert trainer.best_epoch == 1
    assert (tmp_path / "checkpoints" / "best.pt").exists()
