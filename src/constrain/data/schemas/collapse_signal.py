# constrain/data/schemas/collapse_signal.py
import json
from typing import Optional

from pydantic import BaseModel, ConfigDict, field_serializer, field_validator


class CollapseSignalDTO(BaseModel):
    model_config = ConfigDict(from_attributes=True, extra="forbid")
    
    id: Optional[int] = None
    run_id: str
    experiment_id: Optional[int] = None
    
    prediction_horizon: int
    auc_mean: Optional[float] = None
    auc_std: Optional[float] = None
    feature_importance_json: str  # JSON-encoded list
    n_samples: int
    
    mean_energy: Optional[float] = None
    intervention_energy_delta: Optional[float] = None
    created_at: float

    run_id: str
    experiment_id: Optional[int] = None
    
    prediction_horizon: int
    auc_mean: Optional[float] = None
    auc_std: Optional[float] = None
    feature_importance_json: str  # JSON-encoded list
    n_samples: int

    @classmethod
    def parse_json_list(cls, v):
        if isinstance(v, str):
            try:
                return json.loads(v)
            except:
                return []
        return v

    @field_serializer("feature_importance_json")
    def serialize_json_list(self, v):
        return json.dumps(v) if isinstance(v, (list, dict)) else v
