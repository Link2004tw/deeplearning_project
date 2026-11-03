"""Trainer: fit() -> history, saves best checkpoint."""

import math
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, StepLR
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


class Trainer:
    """Training loop with validation, scheduling, early stopping, and checkpointing."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        cfg: Dict[str, Any],
        tracker: Optional[Any] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = dict(cfg)
        self.tracker = tracker
        self.metadata = dict(metadata or {})

        self.device = next(self.model.parameters()).device
        self.model.to(self.device)

        self.epochs = int(self.cfg.get("epochs", 10))
        self.run_name = str(self.cfg.get("run_name", "run"))
        self.output_dir = Path(
            self.cfg.get("output_dir", Path("outputs/checkpoints") / self.run_name)
        )

        es_cfg = self.cfg.get("early_stopping") or {}
        self.monitor = str(es_cfg.get("monitor", self.cfg.get("monitor", "val_acc")))
        self.monitor_mode = "min" if self.monitor in {"val_loss", "loss"} else "max"
        self.patience = int(es_cfg.get("patience", self.cfg.get("patience", 5)))
        self.min_delta = float(es_cfg.get("min_delta", 0.0))
        self.min_epochs = int(es_cfg.get("min_epochs", 0))

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()

        self.use_amp = bool(self.cfg.get("amp", False)) and self.device.type == "cuda"
        self.show_progress = bool(self.cfg.get("progress", True))
        self.scaler = (
            torch.amp.GradScaler("cuda", enabled=True) if self.use_amp else None
        )

        self.best_metric = -math.inf if self.monitor_mode == "max" else math.inf
        self.best_epoch = 0
        self.last_metrics: Dict[str, float] = {}

    def _build_optimizer(self) -> torch.optim.Optimizer:
        name = str(self.cfg.get("optimizer", "adam")).lower()
        lr = float(self.cfg.get("lr", 1e-3))
        weight_decay = float(self.cfg.get("weight_decay", 0.0))

        if name == "adam":
            return torch.optim.Adam(
                self.model.parameters(), lr=lr, weight_decay=weight_decay
            )
        if name == "adamw":
            return torch.optim.AdamW(
                self.model.parameters(), lr=lr, weight_decay=weight_decay
            )
        if name == "sgd":
            return torch.optim.SGD(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                momentum=float(self.cfg.get("momentum", 0.9)),
                nesterov=bool(self.cfg.get("nesterov", False)),
            )
        raise ValueError(f"Unknown optimizer: {name}")

    def _build_scheduler(self):
        name = str(self.cfg.get("scheduler", "none")).lower()
        if name in {"", "none", "null"}:
            return None
        if name == "cosine":
            t_max = max(1, int(self.cfg.get("t_max", self.epochs)))
            return CosineAnnealingLR(
                self.optimizer, T_max=t_max, eta_min=float(self.cfg.get("min_lr", 0.0))
            )
        if name == "step":
            return StepLR(
                self.optimizer,
                step_size=max(1, int(self.cfg.get("lr_step_size", 10))),
                gamma=float(self.cfg.get("lr_gamma", 0.1)),
            )
        if name == "plateau":
            return ReduceLROnPlateau(
                self.optimizer,
                mode=self.monitor_mode,
                factor=float(self.cfg.get("lr_factor", 0.1)),
                patience=int(self.cfg.get("lr_patience", 2)),
                min_lr=float(self.cfg.get("min_lr", 0.0)),
            )
        raise ValueError(f"Unknown scheduler: {name}")

    def _is_improvement(self, value: float) -> bool:
        if self.monitor_mode == "max":
            return value > self.best_metric + self.min_delta
        return value < self.best_metric - self.min_delta

    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        progress = tqdm(
            self.train_loader,
            desc=f"Train {epoch:02d}",
            leave=False,
            disable=not self.show_progress,
        )
        for images, labels in progress:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True).long().reshape(-1)

            self.optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=self.device.type, enabled=self.use_amp):
                logits = self.model(images)
                loss = self.criterion(logits, labels)

            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            total_samples += batch_size
            progress.set_postfix(
                loss=f"{total_loss / max(total_samples, 1):.4f}",
                acc=f"{total_correct / max(total_samples, 1):.4f}",
            )

        total_samples = max(total_samples, 1)
        return {
            "loss": total_loss / total_samples,
            "acc": total_correct / total_samples,
        }

    def _validate_epoch(self) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        all_targets: List[torch.Tensor] = []
        all_preds: List[torch.Tensor] = []

        progress = tqdm(
            self.val_loader,
            desc="Val",
            leave=False,
            disable=not self.show_progress,
        )
        with torch.no_grad():
            for images, labels in progress:
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True).long().reshape(-1)

                with torch.autocast(device_type=self.device.type, enabled=self.use_amp):
                    logits = self.model(images)
                    loss = self.criterion(logits, labels)

                batch_size = labels.size(0)
                total_loss += loss.item() * batch_size
                total_correct += (logits.argmax(dim=1) == labels).sum().item()
                total_samples += batch_size
                all_targets.append(labels.cpu())
                all_preds.append(logits.argmax(dim=1).cpu())
                progress.set_postfix(
                    loss=f"{total_loss / max(total_samples, 1):.4f}",
                    acc=f"{total_correct / max(total_samples, 1):.4f}",
                )

        total_samples = max(total_samples, 1)
        y_true = torch.cat(all_targets).numpy()
        y_pred = torch.cat(all_preds).numpy()
        macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

        return {
            "loss": total_loss / total_samples,
            "acc": total_correct / total_samples,
            "f1": float(macro_f1),
        }

    def _save_checkpoint(self, epoch: int, val_acc: float, is_best: bool = False):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": (
                self.scheduler.state_dict() if self.scheduler is not None else None
            ),
            "epoch": epoch,
            "val_acc": val_acc,
            "best_metric": self.best_metric,
            "best_epoch": self.best_epoch,
            "monitor": self.monitor,
            "config": self.cfg,
            "metadata": self.metadata,
        }

        torch.save(payload, self.output_dir / "last.pt")
        if is_best:
            torch.save(payload, self.output_dir / "best.pt")
            print(f"  saved best checkpoint -> {self.output_dir / 'best.pt'}")
            if self.tracker is not None:
                self.tracker.log_artifact(str(self.output_dir / "best.pt"))

    def fit(self) -> Dict[str, List[float]]:
        history: Dict[str, List[float]] = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "val_f1": [],
            "lr": [],
        }

        epochs_without_improvement = 0

        print(
            f"Training {self.run_name} on {self.device} for {self.epochs} epoch(s) | "
            f"AMP={self.use_amp} | output={self.output_dir}"
        )

        for epoch in range(1, self.epochs + 1):
            train_metrics = self._train_epoch(epoch)
            val_metrics = self._validate_epoch()

            monitor_value = (
                val_metrics["loss"]
                if self.monitor_mode == "min"
                else val_metrics["acc"]
            )

            if isinstance(self.scheduler, ReduceLROnPlateau):
                self.scheduler.step(monitor_value)
            elif self.scheduler is not None:
                self.scheduler.step()

            learning_rate = float(self.optimizer.param_groups[0]["lr"])

            history["train_loss"].append(train_metrics["loss"])
            history["train_acc"].append(train_metrics["acc"])
            history["val_loss"].append(val_metrics["loss"])
            history["val_acc"].append(val_metrics["acc"])
            history["val_f1"].append(val_metrics["f1"])
            history["lr"].append(learning_rate)

            improved = self._is_improvement(monitor_value)
            if improved:
                self.best_metric = monitor_value
                self.best_epoch = epoch
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            self.last_metrics = {
                "epoch": float(epoch),
                "train_loss": train_metrics["loss"],
                "train_acc": train_metrics["acc"],
                "val_loss": val_metrics["loss"],
                "val_acc": val_metrics["acc"],
                "val_f1": val_metrics["f1"],
                "lr": learning_rate,
            }

            self._save_checkpoint(epoch, val_metrics["acc"], is_best=improved)

            if self.tracker is not None:
                self.tracker.log_metrics(self.last_metrics, step=epoch)

            print(
                f"Epoch {epoch:03d}/{self.epochs} | "
                f"train loss {train_metrics['loss']:.4f} acc {train_metrics['acc']:.4f} | "
                f"val loss {val_metrics['loss']:.4f} acc {val_metrics['acc']:.4f} "
                f"f1 {val_metrics['f1']:.4f} | lr {learning_rate:.2e}"
            )

            if (
                self.patience > 0
                and epoch >= self.min_epochs
                and epochs_without_improvement >= self.patience
            ):
                print(
                    f"Early stopping at epoch {epoch} "
                    f"(best {self.monitor}={self.best_metric:.4f} at epoch {self.best_epoch})"
                )
                break

        return history
