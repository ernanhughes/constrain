from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict


class CalibrationDTO(BaseModel):
    model_config = ConfigDict(extra="forbid", from_attributes=True)

    id: Optional[int] = None

    run_id: str
    policy_mode: str

    tau_soft: float
    tau_medium: float
    tau_hard: float

    sample_count: Optional[int] = None

    created_at: float
