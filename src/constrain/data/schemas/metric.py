from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict


class MetricDTO(BaseModel):
    model_config = ConfigDict(extra="forbid", from_attributes=True)

    id: Optional[int] = None
    step_id: int

    stage: str
    metric_name: str
    metric_value: float

    created_at: float
