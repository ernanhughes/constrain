# constrain/experiments/control_experiment_runner.py

from dataclasses import dataclass
from typing import List, Dict
import pandas as pd
import numpy as np
import time
import os

@dataclass
class ControlExperimentConfig:
    seeds: List[int]
    policy_ids: List[int]
    output_dir: str


class ControlExperimentRunner:

    def __init__(self, run_callable, config: ControlExperimentConfig):
        self.run = run_callable  # ← your existing run()
        self.config = config

    def execute(self):

        records = []

        for seed in self.config.seeds:
            for policy_id in self.config.policy_ids:

                print(f"Running seed={seed} policy={policy_id}")

                result = self.run(policy_id=policy_id, seed=seed)

                records.append({
                    "seed": seed,
                    "policy_id": policy_id,
                    "accuracy": result["accuracy"],
                    "final_accuracy": result["final_accuracy"],
                    "collapse_rate": result["collapse_rate"],
                    "mean_energy": result["mean_energy"],
                    "intervention_rate": result["intervention_rate"],
                })

        df = pd.DataFrame(records)
        self._persist(df)
        return self._analyze(df)
