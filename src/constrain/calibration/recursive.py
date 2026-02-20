# constrain/calibration.py

import numpy as np

from constrain.config import get_config


class RecursiveCalibrator:

    MIN_SAMPLES = 20

    @classmethod
    def get_thresholds(cls, memory, run_id):
        """
        Main entry point for recursive threshold resolution.
        Handles:
            - Existing calibration
            - Auto calibration
            - Fallback to static config
        """

        cfg = get_config()

        # 1️⃣ Check existing calibration
        calibration = memory.calibrations.get_latest(
            policy_mode="recursive",
            run_id=run_id,
        )

        if calibration:
            return (
                calibration.tau_soft,
                calibration.tau_medium,
                calibration.tau_hard,
            )

        # 2️⃣ Attempt new calibration
        steps = memory.steps.filter(run_id=run_id)

        if len(steps) < cls.MIN_SAMPLES:
            # Not enough data → fallback to static
            return cfg.tau_soft, cfg.tau_medium, cfg.tau_hard

        energies = np.array([s.total_energy for s in steps])

        tau_soft = float(np.percentile(energies, 60))
        tau_medium = float(np.percentile(energies, 80))
        tau_hard = float(np.percentile(energies, 95))

        # 3️⃣ Persist calibration
        memory.calibrations.create_calibration(
            run_id=run_id,
            policy_mode="recursive",
            tau_soft=tau_soft,
            tau_medium=tau_medium,
            tau_hard=tau_hard,
            sample_count=len(steps),
        )

        return tau_soft, tau_medium, tau_hard


    @staticmethod
    def auto_calibrate(memory, run_id):

        steps = memory.steps.list(filters={"run_id": run_id}, limit=100000)

        energies = np.array([s.total_energy for s in steps])

        if len(energies) < 50:
            raise ValueError("Not enough samples")

        base_soft = np.percentile(energies, 60)
        base_medium = np.percentile(energies, 80)
        base_hard = np.percentile(energies, 95)

        # Look at latest signal report
        report = memory.signal_reports.get_one(
            filters={"run_id": run_id}
        )

        if not report or not report.auc:
            return {
                "tau_soft": float(base_soft),
                "tau_medium": float(base_medium),
                "tau_hard": float(base_hard),
            }

        auc = report.auc

        # Tighten thresholds if predictive power strong
        if auc > 0.75:
            multiplier = 0.9
        elif auc < 0.6:
            multiplier = 1.05
        else:
            multiplier = 1.0

        return {
            "tau_soft": float(base_soft * multiplier),
            "tau_medium": float(base_medium * multiplier),
            "tau_hard": float(base_hard),
        }
