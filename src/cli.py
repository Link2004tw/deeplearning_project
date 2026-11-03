import argparse
import sys
import torch 
import yaml 
from pathlib import Path 
import numpy as np

from src.data import build_dataloaders
from src.models import build_encoder, ImageClassifier
from src.training import Trainer
from src.training.tracking import MLflowTracker
from src.evaluation import evaluate_model, predict_all
from src.visualization import (
    plot_confusion_matrix,
    plot_training_curves,
    plot_model_comparison,
)
from src.visualization.gradcam import GradCAM
from src.visualization.gradcam_report import generate_gradcam_grid, compare_models_gradcam
from src.tuning import SearchSpace, GridSearcher, RandomSearcher, compare_searchers



def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def build_model(config: dict, num_classes: int, device: torch.device):
    model_cfg = config["model"]
    encoder_name = model_cfg.get("encoder") or model_cfg.get("name")
    if not encoder_name:
        raise ValueError("config model section must define 'encoder'")
    encoder = build_encoder(encoder_name, **model_cfg.get("encoder_kwargs", {}))
    model = ImageClassifier(encoder, num_classes=num_classes)
    return model.to(device)

def get_tracker(cfg: dict):
    tracking = cfg.get("tracking", {}) or {}
    backend = tracking.get("backend", "mlflow")
    if str(backend).lower() in {"", "none", "null"}:
        return None
    if backend == "mlflow":
        return MLflowTracker(
            experiment_name=tracking.get("experiment", "cub200"),
            run_name=tracking.get("run_name") or cfg.get("run_name"),
            tracking_uri=tracking.get("tracking_uri"),
        )
    return None

def cmd_train(args):
    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, test_loader, meta = build_dataloaders(cfg["data"])
    num_classes = meta["num_classes"]
    class_names = meta["class_names"]

    model = build_model(cfg, num_classes, device)

    tracker = get_tracker(cfg)
    if tracker: 
        tracker.log_params(cfg)

    trainer = Trainer(model, train_loader, val_loader, cfg["train"], tracker)
    history = trainer.fit()

    if tracker:
        tracker.finish()

    results = evaluate_model(model, val_loader, device, class_names)

    print(f"Validation Accuracy: {results['metrics']['accuracy']:.4f}")
    print(f"Validation Macro F1: {results['metrics']['macro_f1']:.4f}")

    out_dir = Path("outputs/figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_training_curves(history, title=cfg.get("run_name", "Training")).savefig(
        out_dir / f"{cfg.get('run_name', 'train')}_curves.png"
    )
    plot_confusion_matrix(
        results["y_true"], results["y_pred"], class_names,
        save_path=str(out_dir / f"{cfg.get('run_name', 'train')}_confusion.png")
    )


def cmd_evaluate(args):
    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, _, test_loader, meta = build_dataloaders(cfg["data"])
    class_names = meta["class_names"]
    num_classes = meta["num_classes"]

    model = build_model(cfg, num_classes, device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    results = evaluate_model(model, test_loader, device, class_names)
    metrics = results["metrics"]

    print("\n=== Test Set Evaluation ===")
    for k, v in metrics.items():
        if v is not None:
            print(f"  {k}: {v:.4f}")

    print("\n=== Per-Class Report (Top/Bottom 10) ===")
    from src.evaluation.metrics import per_class_report, get_best_worst_classes
    per_class = per_class_report(results["y_true"], results["y_pred"], class_names)
    best, worst = get_best_worst_classes(per_class, n=10)
    print("Best 10:")
    print(best[["class", "f1", "support"]].to_string(index=False))
    print("\nWorst 10:")
    print(worst[["class", "f1", "support"]].to_string(index=False))

    out_dir = Path("outputs/figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_confusion_matrix(
        results["y_true"], results["y_pred"], class_names,
        save_path=str(out_dir / "test_confusion.png")
    )

    if args.gradcam:
        target_layer = model.encoder.cam_target_layer()
        generate_gradcam_grid(
            model, test_loader, device, class_names, target_layer,
            save_path=str(out_dir / "gradcam_grid.png")
        )
        print(f"Grad-CAM grid saved to {out_dir / 'gradcam_grid.png'}")

def cmd_predict(args):
    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, _, _, meta = build_dataloaders(cfg["data"])
    class_names = meta["class_names"]
    num_classes = meta["num_classes"]

    model = build_model(cfg, num_classes, device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
      # Load and preprocess image
    from torchvision import transforms
    from PIL import Image

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    image = Image.open(args.image).convert("RGB")
    img_tensor = transform(image).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        logits = model(img_tensor)
        probs = torch.softmax(logits, dim=1)[0]
        top5 = probs.topk(5)

    print(f"\nTop-5 Predictions for {args.image}:")
    for i, (prob, idx) in enumerate(zip(top5.values, top5.indices)):
        print(f"  {i+1}. {class_names[idx]}: {prob.item():.4f}")

    if args.gradcam:
        target_layer = model.encoder.cam_target_layer()
        gradcam = GradCAM(model, target_layer)
        heatmap = gradcam(img_tensor, class_idx=top5.indices[0].item())

        from src.visualization.gradcam import overlay_heatmap, denormalize_image
        img_np = denormalize_image(np.array(image.resize((224, 224))))
        overlay = overlay_heatmap(img_np, heatmap)

        out_path = Path(args.image).with_suffix(".gradcam.png")
        import cv2
        cv2.imwrite(str(out_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        print(f"Grad-CAM overlay saved to {out_path}")    

def cmd_tune(args):
    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base_cfg = load_config(cfg["base_config"])
    base_cfg["train"]["epochs"] = cfg["budget"]["epochs"]
    if cfg["budget"].get("subset"):
        base_cfg["data"]["subset"] = cfg["budget"]["subset"]

    train_loader, val_loader, _, meta = build_dataloaders(base_cfg["data"])
    num_classes = meta["num_classes"]

    search_space = SearchSpace.from_yaml(args.config)

    tracker = get_tracker(base_cfg)

    def build_and_train(trial_cfg):
        full_cfg = {**base_cfg}
        for k, v in trial_cfg.items():
            if k.startswith("train."):
                full_cfg["train"][k.replace("train.", "")] = v 

        model = build_model(full_cfg, num_classes, device) 
        trainer = Trainer(model, train_loader, val_loader, full_cfg["train"], tracker)
        history = trainer.fit()

        return {
            "val_acc": history["val_acc"][-1],
            "val_f1": history.get("val_f1", [0])[-1],
        }

    if cfg["search"]["method"] == "grid":
        searcher = GridSearcher(search_space, tracker)
        results = searcher.run(base_cfg, build_and_train, output_dir="outputs/tuning")
    else:
        searcher = RandomSearcher(search_space, tracker, seed=cfg["search"].get("seed", 42))
        n_trials = cfg["search"].get("n_trials", 20)
        results = searcher.run(base_cfg, build_and_train, n_trials, output_dir="outputs/tuning")

    grid_path = Path("outputs/tuning/grid_search_trials.csv")
    random_path = Path("outputs/tuning/random_search_trials.csv")
    if grid_path.exists() and random_path.exists():
        from src.tuning.compare import load_trials
        grid_results = load_trials(str(grid_path))
        random_results = load_trials(str(random_path))
        compare_searchers(grid_results, random_results, output_dir="outputs/tuning")
        print("Tuning comparison saved to outputs/tuning/")

    

def main():
    parser = argparse.ArgumentParser(description="CUB-200 Classification Pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Train
    p_train = subparsers.add_parser("train", help="Train a classification model")
    p_train.add_argument("--config", required=True, help="Path to YAML config")
    p_train.set_defaults(func=cmd_train)

    # Evaluate
    p_eval = subparsers.add_parser("evaluate", help="Evaluate a trained model")
    p_eval.add_argument("--config", required=True, help="Path to YAML config")
    p_eval.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    p_eval.add_argument("--gradcam", action="store_true", help="Generate Grad-CAM grid")
    p_eval.set_defaults(func=cmd_evaluate)

    # Predict
    p_pred = subparsers.add_parser("predict", help="Predict on a single image")
    p_pred.add_argument("--config", required=True, help="Path to YAML config")
    p_pred.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    p_pred.add_argument("--image", required=True, help="Path to input image")
    p_pred.add_argument("--gradcam", action="store_true", help="Generate Grad-CAM overlay")
    p_pred.set_defaults(func=cmd_predict)

    # Tune
    p_tune = subparsers.add_parser("tune", help="Run hyperparameter tuning")
    p_tune.add_argument("--config", required=True, help="Path to tuning YAML config")
    p_tune.set_defaults(func=cmd_tune)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":    
    main()