from __future__ import annotations
from dataclasses import dataclass

from constrain.calibration.recursive import RecursiveCalibrator
from constrain.data.memory import Memory
from constrain.policy.log_calibrator import LogCalibrator
from .custom_types import ThresholdProvider, Thresholds
from .threshold_query import ThresholdQuery


@dataclass
class DatabaseThresholdProvider(ThresholdProvider):
    """
    Thresholds derived strictly from persisted logs via stores.
    No direct SQL usage in core.
    """

    query: ThresholdQuery

    def get(self, *, cfg, memory: Memory, run_id: str, step_id: int | None = None) -> Thresholds:
        # Allow query to default to the current run if not specified
        q = self.query
        if q.run_id is None and q.include_run_ids is None and q.last_n_runs is None:
            q = ThresholdQuery(**{**q.__dict__, "run_id": run_id})
        return memory.steps.get_energy_thresholds(q)

@dataclass
class StaticThresholdProvider(ThresholdProvider):
    def get(self, *, cfg, memory: Memory, run_id: str, step_id: int | None = None) -> Thresholds:
        return Thresholds(cfg.tau_soft, cfg.tau_medium, cfg.tau_hard)

@dataclass
class RecursiveThresholdProvider(ThresholdProvider):
    calibrator: type[RecursiveCalibrator] = RecursiveCalibrator

    def get(self, *, cfg, memory: Memory, run_id: str, step_id: int | None = None) -> Thresholds:
        tau_soft, tau_medium, tau_hard = self.calibrator.get_thresholds(memory, run_id)
        return Thresholds(tau_soft, tau_medium, tau_hard)
    

from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class CalibrationThresholdProvider(ThresholdProvider):
    policy_mode: str = "dynamic"

    def get(self, *, cfg, memory: Memory, run_id: str, step_id=None):

        # -----------------------------------------------------
        # 1️⃣ Try existing calibration record for this run
        # -----------------------------------------------------
        calibration = memory.calibrations.get_latest(
            policy_mode=self.policy_mode,
            run_id=run_id,
        )

        if calibration is not None:
            return Thresholds(
                tau_soft=calibration.tau_soft,
                tau_medium=calibration.tau_medium,
                tau_hard=calibration.tau_hard,
            )

        # -----------------------------------------------------
        # 2️⃣ Try calibrating current run
        # -----------------------------------------------------
        calibrator = LogCalibrator()

        try:
            return calibrator.calibrate(memory=memory, run_id=run_id)

        except ValueError as e:
            logger.warning(
                "Calibration failed for current run %s (%s). Trying fallback.",
                run_id,
                str(e),
            )

        # -----------------------------------------------------
        # 3️⃣ Try previous completed run
        # -----------------------------------------------------
        try:
            previous_run_id = memory.runs.get_previous_completed_run(run_id)

            if previous_run_id:
                logger.info(
                    "Using thresholds from previous run: %s",
                    previous_run_id,
                )
                return calibrator.calibrate(
                    memory=memory,
                    run_id=previous_run_id,
                )

        except Exception as e:
            logger.warning(
                "Previous run calibration failed: %s",
                str(e),
            )

        # -----------------------------------------------------
        # 4️⃣ Try global calibration (no run filter)
        # -----------------------------------------------------
        try:
            logger.info("Attempting global calibration fallback.")
            return calibrator.calibrate(memory=memory, run_id=None)

        except Exception as e:
            logger.warning(
                "Global calibration failed: %s",
                str(e),
            )

        # -----------------------------------------------------
        # 5️⃣ Final fallback: static config
        # -----------------------------------------------------
        logger.warning(
            "Falling back to static config thresholds."
        )

        return Thresholds(
            tau_soft=cfg.tau_soft,
            tau_medium=cfg.tau_medium,
            tau_hard=cfg.tau_hard,
        )