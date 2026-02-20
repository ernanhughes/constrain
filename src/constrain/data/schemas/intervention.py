# constrain/data/schemas/intervention.py

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict


class InterventionDTO(BaseModel):
    model_config = ConfigDict(extra="forbid", from_attributes=True)

    id: Optional[int] = None

    run_id: str
    problem_id: int
    iteration: int

    threshold: str          # "soft" | "medium" | "hard"
    rationale: Optional[str] = None
    reverted_to: Optional[int] = None

    new_temperature: Optional[float] = None

    timestamp: float
