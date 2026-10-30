import pytest
import numpy as np 

from src.tuning.search_space import SearchSpace, ParamSpec, ParamType
from src.tuning.grid_search import GridSearcher
from src.tuning.random_search import RandomSearcher
from src.tuning.compare import TrialResult, compare_searchers


class DummyTrainer: 
    def __init__(self, seed=42):
        self.rng = np.random.default_rng(seed)
        self.call_count = 0

    def __call__(self, cfg):
        self.call_count += 1
        lr = cfg.get("lr", 1e-3)
        batch = cfg.get("batch_size", 32)
        wd = cfg.get("weight_decay", 1e-4)
        opt = cfg.get("optimizer", "adam")

        val_acc = 0.5 + 0.3 * (1 - np.log10(lr) / 5) + 0.05 * (batch / 64) - 0.1 * wd
        if opt == "adam":
            val_acc += 0.02
        val_acc = np.clip(val_acc, 0.3, 0.95)
        val_f1 = val_acc * 0.95

        return {
            "val_acc": float(val_acc),
            "val_f1": float(val_f1),
            "time_s": float(self.rng.uniform(10, 30)),
        }


def test_search_space_from_yaml(tmp_path):
    yaml_content = """
lr:
  type: log_uniform
  low: 0.00001
  high: 0.01
batch_size:
  type: categorical
  values: [32, 64]
"""
    path = tmp_path / "test_space.yaml"
    path.write_text(yaml_content)

    space = SearchSpace.from_yaml(str(path))
    assert "lr" in space.params
    assert "batch_size" in space.params
    assert space.params["lr"].param_type == ParamType.LOG_UNIFORM
    assert space.params["batch_size"].values == [32, 64]


def test_grid_search_generates_all_combinations():
    space = SearchSpace({
        "lr": ParamSpec(ParamType.FLOAT, low=0.001, high=0.01, step=0.003),
        "batch_size": ParamSpec(ParamType.CATEGORICAL, values=[32, 64]),
    })
    configs = space.grid_configs()
    assert len(configs) == 4 * 2  # 4 lr values * 2 batch sizes = 8 combinations
    lrs = [c["lr"] for c in configs]
    batches = [c["batch_size"] for c in configs]
    # Each lr appears twice (once per batch_size)
    unique_lrs = sorted(set(round(lr, 10) for lr in lrs))
    assert unique_lrs == [0.001, 0.004, 0.007, 0.01]
    assert set(batches) == {32, 64}


def test_random_search_reproducible():
    space = SearchSpace({
        "lr": ParamSpec(ParamType.LOG_UNIFORM, low=1e-5, high=1e-2),
        "batch_size": ParamSpec(ParamType.CATEGORICAL, values=[32, 64]),
    })

    searcher1 = RandomSearcher(space, seed=42)
    searcher2 = RandomSearcher(space, seed=42)

    cfg1 = searcher1.search_space.sample_config(searcher1._rng)
    cfg2 = searcher2.search_space.sample_config(searcher2._rng)
    assert cfg1 == cfg2


def test_grid_searcher_runs(tmp_path):
    space = SearchSpace({
        "lr": ParamSpec(ParamType.CATEGORICAL, values=[0.001, 0.01]),
        "batch_size": ParamSpec(ParamType.CATEGORICAL, values=[32]),
    })
    searcher = GridSearcher(space)
    trainer = DummyTrainer()

    results = searcher.run(
        base_cfg={"epochs": 1},
        build_and_train_fn=trainer,
        output_dir=str(tmp_path / "tuning"),
    )

    assert len(results) == 2
    assert all(isinstance(r, TrialResult) for r in results)
    assert trainer.call_count == 2


def test_random_searcher_runs(tmp_path):
    space = SearchSpace({
        "lr": ParamSpec(ParamType.LOG_UNIFORM, low=1e-4, high=1e-2),
        "batch_size": ParamSpec(ParamType.CATEGORICAL, values=[32, 64]),
    })
    searcher = RandomSearcher(space, seed=123)
    trainer = DummyTrainer()

    results = searcher.run(
        base_cfg={"epochs": 1},
        build_and_train_fn=trainer,
        n_trials=5,
        output_dir=str(tmp_path / "tuning"),
    )

    assert len(results) == 5
    assert trainer.call_count == 5


def test_compare_searchers(tmp_path):
    grid_results = [
        TrialResult(0, "grid", {"lr": 0.001}, {"val_acc": 0.7, "val_f1": 0.65, "time_s": 10}),
        TrialResult(1, "grid", {"lr": 0.01}, {"val_acc": 0.8, "val_f1": 0.75, "time_s": 12}),
    ]
    random_results = [
        TrialResult(0, "random", {"lr": 0.005}, {"val_acc": 0.75, "val_f1": 0.7, "time_s": 11}),
        TrialResult(1, "random", {"lr": 0.002}, {"val_acc": 0.85, "val_f1": 0.8, "time_s": 13}),
    ]

    summary = compare_searchers(grid_results, random_results, output_dir=str(tmp_path / "tuning"))

    assert "grid" in summary and "random" in summary
    assert summary["grid"]["best_score"] == 0.8
    assert summary["random"]["best_score"] == 0.85
    assert (tmp_path / "tuning" / "tuning_comparison.csv").exists()
    assert (tmp_path / "tuning" / "tuning_comparison.png").exists()
    assert (tmp_path / "tuning" / "tuning_comparison.json").exists()