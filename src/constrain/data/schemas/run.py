# constrain/data/schemas/run.py

from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict


class RunDTO(BaseModel):
    model_config = ConfigDict(extra="forbid", from_attributes=True)

    run_id: str

    model_name: str
    initial_temperature: float

    num_problems: int
    num_recursions: int

    tau_soft: float
    tau_medium: float
    tau_hard: float

    policy_id: int
    task_type: str

    start_time: float
    end_time: Optional[float] = None

    status: str = "running"
    notes: Optional[str] = None

    tau_soft_calibrated: Optional[float] = None
    tau_medium_calibrated: Optional[float] = None
    tau_hard_calibrated: Optional[float] = None
