from typing import Dict, Any, List, Callable, Optional 
from dataclasses import dataclass, asdict
import time 
import csv
import os 
import numpy as np
from .search_space import SearchSpace
from .compare import TrialResult 

class RandomSearcher: 

    def __init__(
            self,
            search_space: SearchSpace,
            tracker: Optional[Any] = None,
            seed: int = 42
            ):
        self.search_space = search_space
        self.tracker = tracker
        self.seed = seed
        self._rng = np.random.default_rng(seed)

    def run(
            self, 
            base_cfg: Disct[str, Any],
            build_and_train_fn: Callable[[Dict[str, Any]], Dict[str, float]],
            n_trials: int,
            output_dir: str = "outputs/tuning",
    ) -> List[TrialResult]:
        os.makedirs(output_dir, exist_ok=True)
        csv_path = os.path.join(output_dir, "random_search_trials.csv")
        fieldnames = ["trial_id", "search_type", "val_acc", "val_f1", "time_s"] + list(self.search_space.params.keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

        results = []
        for trial_id in range(n_trials):
            trial_params = self.search_space.sample_config(self._rng)
            cfg = {**base_cfg, **trial_params}

            if self.tracker:
                self.tracker.log_params({**cfg, "search": "random", "trial_id": trial_id})

            start_time = time.time()
            try: 
                metrics = build_and_train_fn(cfg)
            except Exception as e:
                metrics = {"val_acc": 0.0, "val_f1": 0.0, "time_s": time.time() - start_time}
                if self.tracker:
                    self.tracker.log_metrics({"error": str(e)}, step=trial_id)

                elapsed = time.time() - start_time
                metrics.setdefault("time_s", elapsed)

                if self.tracker: 
                    self.tracker.log_metrics(
                        {**metrics, "search": "random", "trial_id": trial_id},
                    step=trial_id
                    )

                results.append(TrialResult(trial_id=trial_id,
                search_type="random",
                params=trial_params,
                metrics=metrics))

                row = {
                "trial_id": trial_id,
                "search_type": "random",
                "val_acc": metrics.get("val_acc", 0.0),
                "val_f1": metrics.get("val_f1", 0.0),
                "time_s": metrics.get("time_s", elapsed),
                **trial_params,
            }
            with open(csv_path, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writerow(row)
        return results
