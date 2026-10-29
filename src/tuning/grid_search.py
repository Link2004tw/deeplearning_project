from typing import Dict, Any, List, Callable, Optional 
from dataclasses import dataclass, asdict
import time 
import csv
import os 
import numpy as np
from .search_space import SearchSpace
from .compare import TrialResult 


@dataclass
class GridSearcher:
    search_space: SearchSpace
    tracker: Optional[Any] = None 

    def run(self,
        base_cfg: Dict[str, Any],
        build_and_train_fn: Callable[[Dict[str, Any]], Dict[str, float]],
        n_trials: Optional[int] = None,
        output_dir: str = "outputs/tuning")-> List[TrialResult]:

        os.makedirs(output_dir, exist_ok=True)
        csv_path = os.path.join(output_dir, "grid_search_trials.csv")

        # Initialize CSV with header
        fieldnames = ["trial_id", "search_type", "val_acc", "val_f1", "time_s"] + list(self.search_space.params.keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

        configs = self.search_space.grid_configs()
        if n_trials is not None:
            configs = configs[:n_trials]

        results = []
        for trial_id, trial_params in enumerate(configs):
            # Merge base config with trial params
            cfg = {**base_cfg, **trial_params}

            # Track trial
            if self.tracker:
                self.tracker.log_params({**cfg, "search": "grid", "trial_id": trial_id})

            # Run training
            start_time = time.time()
            try:
                metrics = build_and_train_fn(cfg)
            except Exception as e:
                metrics = {"val_acc": 0.0, "val_f1": 0.0, "time_s": time.time() - start_time}
                if self.tracker:
                    self.tracker.log_metrics({"error": str(e)}, step=trial_id)

            elapsed = time.time() - start_time
            metrics.setdefault("time_s", elapsed)

            # Log metrics
            if self.tracker:
                self.tracker.log_metrics(
                    {**metrics, "search": "grid", "trial_id": trial_id},
                    step=trial_id
                )

            # Record result
            result = TrialResult(
                trial_id=trial_id,
                search_type="grid",
                params=trial_params,
                metrics=metrics,
            )
            results.append(result)

            # Append to CSV immediately (crash-safe)
            row = {
                "trial_id": trial_id,
                "search_type": "grid",
                "val_acc": metrics.get("val_acc", 0.0),
                "val_f1": metrics.get("val_f1", 0.0),
                "time_s": metrics.get("time_s", elapsed),
                **trial_params,
            }
            with open(csv_path, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writerow(row)

        return results
