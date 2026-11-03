# From Pixels to Language: CUB-200-2011 Classification, Colorization and Captioning

Course project for CSE 485 / 488 Deep Learning.

**Status:** Milestone 1 classification baseline complete. Handoff results and lightweight model weights are in [`handoff/`](handoff/README.md).

## Setup

```powershell
python -m venv .venv
& ".venv\Scripts\Activate.ps1"
pip install -r requirements.txt
```

Linux/macOS activation: `source .venv/bin/activate`.

The first CUB run downloads about 1.1 GB:

```powershell
python -c "from src.data.load_data import load_data; tr, te = load_data('data/cub'); print(len(tr), len(te))"
```

Expected: `5994 5794`.

## Training

```powershell
python -m src.cli train --config configs/cls_scratch.yaml
python -m src.cli train --config configs/cls_resnet.yaml
```

Windows note: both configs use `num_workers: 0` to avoid worker-spawn memory errors. Checkpoints are written to `outputs/checkpoints/<run_name>/best.pt` and `last.pt`.

## Evaluation

```powershell
python -m src.cli evaluate --config configs/cls_scratch.yaml --checkpoint outputs/checkpoints/scratch_cnn/best.pt
python -m src.cli evaluate --config configs/cls_resnet.yaml --checkpoint outputs/checkpoints/resnet18/best.pt
```

Evaluation uses the test split and prints accuracy, top-5 accuracy, macro-F1, and per-class results. It writes `outputs/figures/test_confusion.png`.

## Prediction

```powershell
python -m src.cli predict --config configs/cls_resnet.yaml --checkpoint outputs/checkpoints/resnet18/best.pt --image path/to/image.jpg
```

Add `--gradcam` to generate a Grad-CAM overlay.

## MLflow

```powershell
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Open `http://127.0.0.1:5000`, then use **Experiments** -> `cub200-classification`. The **Traces** tab is for LLM tracing and will be empty for training runs. `mlflow.db` and `mlruns/` are local and not committed.

## Milestone 1 Results

| Model | Best epoch | Val acc | Test acc | Test top-5 |
|---|---:|---:|---:|---:|
| Scratch CNN | 19 | 7.67% | 7.06% | 24.20% |
| ResNet18 | 13 | 72.00% | 71.49% | 91.32% |

See [`handoff/README.md`](handoff/README.md) for run IDs, test metrics, configs, logs, figures, and inference-only weights.

## Layout

```
configs/            run configs (classification + tuning)
handoff/            Milestone 1 handoff bundle (metrics, weights, figures, logs)
notebooks/          experiments only, no library code
src/data/           dataset, transforms, EDA
src/models/         BaseEncoder, ImageClassifier, encoder registry
src/training/       Trainer and experiment tracking
src/evaluation/     prediction and metrics
src/tuning/         grid and random search
src/visualization/  plots, Grad-CAM
src/cli.py          CLI entry point
tests/              fakes and unit tests
outputs/            figures, checkpoints, logs, tuning results (not committed)
```
