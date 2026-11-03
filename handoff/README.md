# Milestone 1 Handoff - Classification Models

Person B handoff for the CUB-200-2011 classification/encoder phase.

## Results

| Model | Params | Best epoch | Val acc | Val macro-F1 | Test acc | Test top-5 | Test macro-F1 |
|---|---|---:|---:|---:|---:|---:|---:|
| Scratch CNN | 128 px, batch 32 | 19 | 7.67% | 5.58% | 7.06% | 24.20% | 5.23% |
| ResNet18 | 224 px, batch 16 | 13 | 72.00% | 71.42% | 71.49% | 91.32% | 71.56% |

The scratch CNN is clearly underfit after 20 epochs. ResNet18 improves strongly, then overfits after epoch 13: final training accuracy was 99.48%, while final validation accuracy was 71.67%. Use `best.pt`, not the last epoch, for evaluation and Grad-CAM.

## MLflow

The local MLflow store is `mlflow.db`. Start the UI with:

```powershell
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Open `http://127.0.0.1:5000` and use **Experiments**, not **Traces**. `Traces` is for LLM tracing and will be empty for these runs.

Experiment: `cub200-classification`

| Run | Run ID |
|---|---|
| ResNet18 | `e85178d0e6a94b47b6b3933a55c690db` |
| Scratch CNN | `5f3c3c6c979c47308146fb7dd10ec856` |

`mlflow_runs.csv` contains the exported run summary. The `mlflow.db` and `mlruns/` directories are intentionally not committed.

## Checkpoints

Lightweight inference checkpoints are included here:

- `weights/scratch_cnn_best_state_dict.pt` (2.4 MB)
- `weights/resnet18_best_state_dict.pt` (43.1 MB)

These contain `model_state_dict`, the full config, class names, image size, encoder name, and best-epoch metadata. They intentionally exclude optimizer and scheduler state, so they are for evaluation/inference, not resuming training.

Full resumable checkpoints remain local:

- `outputs/checkpoints/scratch_cnn/best.pt` (7.2 MB)
- `outputs/checkpoints/resnet18/best.pt` (129.2 MB)

The ResNet18 full checkpoint exceeds GitHub's 100 MB file limit, which is why the lightweight state-dict version is included.

## How to evaluate the handoff weights

From the repository root:

```powershell
& ".venv\Scripts\python.exe" -m src.cli evaluate --config handoff/configs/cls_scratch.yaml --checkpoint handoff/weights/scratch_cnn_best_state_dict.pt
& ".venv\Scripts\python.exe" -m src.cli evaluate --config handoff/configs/cls_resnet.yaml --checkpoint handoff/weights/resnet18_best_state_dict.pt
```

The evaluation command uses the test split and prints accuracy, top-5 accuracy, macro-F1, and per-class best/worst results. It writes `outputs/figures/test_confusion.png`; rename or copy it after each run because the path is overwritten.

## What is in this folder

- `metrics.csv` - best and final validation metrics plus test metrics for both models.
- `mlflow_runs.csv` - exported MLflow run summary.
- `configs/` - exact copies of the training configs.
- `weights/` - lightweight best model state dicts.
- `figures/` - training curves and validation/test confusion matrices.
- `logs/` - full training/evaluation logs and cleaned per-epoch summaries.

## Training setup

- Dataset: `Multimodal-Fatima/CUB_train` (5,994 images) and `CUB_test` (5,794 images), 200 classes.
- Validation: 10% stratified split from the training set.
- Data check: train and test filenames have zero overlap; train filenames are unique.
- Windows: `num_workers: 0` avoids worker-spawn memory errors.
- AMP enabled on CUDA.
- Scratch CNN: 128 px, batch 32, Adam, lr 1e-3, 20 epochs, cosine schedule.
- ResNet18: 224 px, batch 16, AdamW, lr 3e-4, 15 epochs, full fine-tune, cosine schedule.

## Important notes

- `cmd_train` prints validation metrics from the last in-memory epoch. The reported handoff metrics use `best.pt`.
- Test confusion matrices are included as `*_test_confusion.png`.
- The stale MLflow run `f5d20cf625b84e708eb6a4357c36def8` is from the failed Windows worker-spawn smoke test and should be ignored.
- The scratch baseline should be improved in a later iteration if time allows; the current comparison mainly demonstrates the benefit of transfer learning.
