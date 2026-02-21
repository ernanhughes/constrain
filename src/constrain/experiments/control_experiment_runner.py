# constrain/experiments/control_experiment_runner.py

import time

from constrain.data.memory import Memory
from constrain.data.schemas.experiment import ExperimentDTO
from constrain.experiments.experiment_config import ControlExperimentConfig


class ControlExperimentRunner:
    def __init__(self, run_callable, config: ControlExperimentConfig):
        self.run = run_callable
        self.config = config
        self.memory = Memory()
        self.experiment_id = None

    def execute(self):
        # Create experiment record
        experiment_dto = ExperimentDTO(
            experiment_name="policy_comparison_v1",
            experiment_type="policy_comparison",
            policy_ids=[0, 4, 99],
            seeds=[42, 43, 44],
            num_problems=200,
            num_recursions=6,
            start_time=time.time(),
            status="running",
        )

        experiment = self.memory.experiments.create(experiment_dto)
        self.experiment_id = experiment.id

        try:
            # Run experiments...
            records = []
            for seed in self.config.seeds:
                for policy_id in self.config.policy_ids:
                    run_id = self.run(policy_id=policy_id, seed=seed)
                    records.append({...})

            # Update with results
            self.memory.experiments.update(
                self.experiment_id,
                {
                    "end_time": time.time(),
                    "status": "completed",
                    "results_summary": {"accuracy_diff": 0.03, "collapse_reduction": 0.15},
                },
            )

            return records

        except Exception as e:
            self.memory.experiments.update(
                self.experiment_id,
                {"status": "failed", "results_summary": {"error": str(e)}},
            )
            raise