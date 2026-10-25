# From Pixels to Language: CUB-200-2011 Classification, Colorization and Captioning

Course project for CSE 485 / 488 Deep Learning.

**Status:** Milestone 1 skeleton (folders and modules stubbed, contracts frozen in `plan.md`).

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Layout

```
configs/            run configs (classification + tuning)
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

## Commands

```bash
python -m src.cli train    --task classification --config configs/cls_scratch.yaml
python -m src.cli evaluate --task classification --checkpoint outputs/checkpoints/<run>/best.pt --split test
python -m src.cli predict  --task classification --checkpoint <path> --image path/to/bird.jpg [--gradcam]
python -m src.cli tune     --task classification --config configs/tuning_random.yaml
```

## Interfaces

See `plan.md` section 3 for the frozen data, model, training, tracking and evaluation contracts.
