from typing import Dict, List, Any, Optional
from dataclasses import dataclass 
import pandas as pd 
import numpy as np 
import matplotlib 
matplotlib.use("Agg")
import matplotlib.pyplot as plt 
import os 
import json 

@dataclass
class TrialResult: 
    trial_id: int 
    search_type: str # ya grid ya random
    params: Dict[str, Any]
    metrics: Dict[str, float]

    def to_dict(self) -> Dict: 
        return {
            "trial_id": self.trial_id,
            "search_type": self.search_type,
            "params": self.params,
        }

def load_trials(csv_path: str) -> List[TrialResult]:
    df = pd.read_csv(csv_path)
    trials = []
    param_cols = [c for c in df.columns if c not in ["trial_id", "search_type", "val_acc", "val_f1", "time_s"]]
    for _, row in df.iterrows():
        params = {
            col: row[col] for col in param_cols
        } 
        metrics = {
            "val_acc": row["val_acc"],
            "val_f1": row["val_f1"],
            "time_s": row["time_s"],
        }
        trials.append(TrialResult(
            trial_id=int(row["trial_id"]),
            search_type=row["search_type"],
            params=params,
            metrics=metrics,
        ))

    return trials 

def compare_searchers(
        grid_results: List[TrialResult],
        random_results: List[TrialResult],
        output_dir: str = "outputs/tuning",
        metric: str = "val_acc"
        ) -> Dict[str, Any]:
    os.makedirs(output_dir, exist_ok=True)

    grid_df = pd.DataFrame([r.to_dict() for r in grid_results])
    random_df = pd.DataFrame([r.to_dict() for r in random_results])

    grid_metrics = pd.json_normalize(grid_df["metrics"])
    random_metrics = pd.json_normalize(random_df["metrics"])

    grid_df = pd.concat([grid_df.drop(columns=["metrics"]), grid_metrics], axis=1)
    random_df = pd.concat([random_df.drop(columns=["metrics"]), random_metrics], axis=1)

    summary = {
        "grid": {
            "n_trials": len(grid_df),
        "best_score": float(grid_df[metric].max()),
        "mean_score": float(grid_df[metric].mean()),
        "std_score": float(grid_df[metric].std()),
        "total_time_s": float(grid_df["time_s"].sum()),
        "mean_time_s": float(grid_df["time_s"].mean()),
        },
        "random": {
            "n_trials": len(random_df),
        "best_score": float(random_df[metric].max()),
        "mean_score": float(random_df[metric].mean()),
        "std_score": float(random_df[metric].std()),
        "total_time_s": float(random_df["time_s"].sum()),
        "mean_time_s": float(random_df["time_s"].mean()),
        }
    }
    best_grid_idx = grid_df[metric].idxmax()
    best_random_idx = random_df[metric].idxmax()
    summary["grid"]["best_params"] = grid_df.loc[best_grid_idx, "params"]
    summary["random"]["best_params"] = random_df.loc[best_random_idx, "params"]

    # Save summary JSON
    with open(os.path.join(output_dir, "tuning_comparison.json"), "w") as f:
        json.dump(summary, f, indent=2)

    comparison_df = pd.DataFrame({
    "method": ["grid", "random"],
    "n_trials": [summary["grid"]["n_trials"], summary["random"]["n_trials"]],
    f"best_{metric}": [summary["grid"]["best_score"], summary["random"]["best_score"]],
    f"mean_{metric}": [summary["grid"]["mean_score"], summary["random"]["mean_score"]],
    "total_time_s": [summary["grid"]["total_time_s"], summary["random"]["total_time_s"]],
    "mean_time_s": [summary["grid"]["mean_time_s"], summary["random"]["mean_time_s"]],
    })
    comparison_df.to_csv(os.path.join(output_dir, "tuning_comparison.csv"), index=False)

    # Plot 1: Best score vs trial number
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Grid: cumulative best
    grid_cummax = grid_df[metric].cummax()
    axes[0].plot(range(1, len(grid_cummax) + 1), grid_cummax, marker="o", label="Grid", color="tab:blue")
    random_cummax = random_df[metric].cummax()
    axes[0].plot(range(1, len(random_cummax) + 1), random_cummax, marker="s", label="Random", color="tab:orange")
    axes[0].set_xlabel("Trial Number")
    axes[0].set_ylabel(f"Best {metric} so far")
    axes[0].set_title("Best Score vs Trial Number")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Score distribution
    axes[1].hist(grid_df[metric], alpha=0.5, label="Grid", bins=10, density=True, color="tab:blue")
    axes[1].hist(random_df[metric], alpha=0.5, label="Random", bins=10, density=True, color="tab:orange")
    axes[1].set_xlabel(metric)
    axes[1].set_ylabel("Density")
    axes[1].set_title("Score Distribution")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "tuning_comparison.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Plot 3: Parameter importance (for random search - correlation with metric)
    if len(random_df) > 5:
        param_cols = [c for c in random_df.columns if c not in ["trial_id", "search_type", "val_acc", "val_f1", "time_s", "params"]]
        if param_cols:
            fig, axes = plt.subplots(1, len(param_cols), figsize=(4 * len(param_cols), 4))
            if len(param_cols) == 1:
                axes = [axes]
            for ax, param in zip(axes, param_cols):
                ax.scatter(random_df[param], random_df[metric], alpha=0.6)
                ax.set_xlabel(param)
                ax.set_ylabel(metric)
                ax.set_title(f"{param} vs {metric}")
                ax.grid(True, alpha=0.3)
            plt.tight_layout()
            fig.savefig(os.path.join(output_dir, "param_importance.png"), dpi=200, bbox_inches="tight")
            plt.close(fig)

    return summary


