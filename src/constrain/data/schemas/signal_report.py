from typing import Optional

from pydantic import BaseModel, ConfigDict


class SignalReportDTO(BaseModel):
    model_config = ConfigDict(extra="forbid", from_attributes=True)

    id: Optional[int] = None
    run_id: str

    mean_energy: Optional[float]
    mean_energy_slope: Optional[float]
    intervention_recovery_delta: Optional[float]

    auc: Optional[float]

    feature_importance_json: Optional[str]

    created_at: float
