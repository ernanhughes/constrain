from dataclasses import dataclass
from constrain.policy.custom_types import Thresholds
from constrain.policy.threshold_query import ThresholdQuery
from constrain.data.memory import Memory


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
            run_id=None,                   # NEVER current run
            last_n_runs=self.last_n_runs,  # use recent runs
            last_n_steps=self.last_n_steps,
            q_soft=self.q_soft,
            q_medium=self.q_medium,
            q_hard=self.q_hard,
            min_samples=self.min_samples,
        )

        result = memory.steps.get_energy_thresholds(query)

        # Only persist calibration if run_id is real
        if run_id is not None:
            memory.calibrations.create_calibration(
                run_id=run_id,
                policy_mode=policy_mode,
                tau_soft=result.thresholds.tau_soft,
                tau_medium=result.thresholds.tau_medium,
                tau_hard=result.thresholds.tau_hard,
                sample_count=result.sample_count,
            )

        return result.thresholds
