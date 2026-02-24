from dataclasses import dataclass

from constrain.policy.custom_types import Thresholds


@dataclass
class ThresholdResolutionService:

    def resolve_and_persist(
        self,
        *,
        provider,
        memory,
        cfg,
        run_id,
        policy_mode="dynamic",
    ) -> Thresholds:

        thresholds = provider.get(
            cfg=cfg,
            memory=memory,
            run_id=run_id,
        )

        memory.calibrations.create_calibration(
            run_id=run_id,
            policy_mode=policy_mode,
            tau_soft=thresholds.tau_soft,
            tau_medium=thresholds.tau_medium,
            tau_hard=thresholds.tau_hard,
            sample_count=None,
        )

        return thresholds