from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict


class RunDTO(BaseModel):
    model_config = ConfigDict(extra="forbid", from_attributes=True)

    run_id: str

    model_name: str
    initial_temperature: float

    num_problems: int
    num_recursions: int

    # Energy thresholds
    tau_soft: float
    tau_medium: float
    tau_hard: float

    tau_soft_calibrated: Optional[float] = None
    tau_medium_calibrated: Optional[float] = None
    tau_hard_calibrated: Optional[float] = None

    # Risk thresholds (hybrid collapse)
    risk_tau_stable: Optional[float] = None
    risk_tau_unstable: Optional[float] = None

    policy_id: int
    task_type: str

    # Survival summary
    mean_survival_time: Optional[float] = None
    collapse_rate: Optional[float] = None
    hazard_ratio_vs_baseline: Optional[float] = None

    # DR summary (doubly robust) How do I test this OK here's my runner how do I test this
    dr_ate: Optional[float] = None
    dr_ci_lower: Optional[float] = None
    dr_ci_upper: Optional[float] = None

    start_time: float
    end_time: Optional[float] = None

    status: str = "running"
    notes: Optional[str] = None
    seed: Optional[int] = None