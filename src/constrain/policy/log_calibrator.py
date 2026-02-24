from dataclasses import dataclass

from constrain.data.memory import Memory
from constrain.policy.custom_types import Thresholds
from constrain.policy.threshold.threshold_query import ThresholdQuery


@dataclass
class LogCalibrator:
    last_n_steps: int = 50000
    last_n_runs: int = 50
    min_samples: int = 20

    # quantiles (recommended defaults)
    q_soft: float = 0.80
    q_medium: float = 0.90
    q_hard: float = 0.97

    def calibrate(
        self,
        *,
        memory: Memory,
        run_id: str | None,
        policy_mode: str = "dynamic",
    ) -> Thresholds:

        query = ThresholdQuery(
            run_id=None,
            exclude_run_ids=[run_id],  # add this
            last_n_runs=self.last_n_runs,
            last_n_steps=self.last_n_steps,
            q_soft=self.q_soft,
            q_medium=self.q_medium,
            q_hard=self.q_hard,
            min_samples=self.min_samples,
        )

        result = memory.steps.get_energy_thresholds(query)

        return result.thresholds

    def persist_calibration(
        self,
        *,
        memory: Memory,
        run_id: str,
        policy_mode: str,
        thresholds: Thresholds,
        sample_count: int,
    ):
        memory.calibrations.create_calibration(
            run_id=run_id,
            policy_mode=policy_mode,
            tau_soft=thresholds.tau_soft,
            tau_medium=thresholds.tau_medium,
            tau_hard=thresholds.tau_hard,
            sample_count=sample_count,
        )