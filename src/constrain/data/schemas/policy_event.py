# constrain/data/schemas/policy_event.py

from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, ConfigDict


class PolicyEventDTO(BaseModel):
    model_config = ConfigDict(extra="forbid", from_attributes=True)

    id: Optional[int] = None

    run_id: str
    step_id: int
    policy_id: int

    tau_soft: float
    tau_medium: float
    tau_hard: float

    action: str
    collapse_probability: Optional[float] = None

    created_at: float