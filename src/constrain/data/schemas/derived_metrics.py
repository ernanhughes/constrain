# constrain/data/schemas/derived_metrics.py

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict


class DerivedMetricsDTO(BaseModel):
    model_config = ConfigDict(extra="forbid", from_attributes=True)

    id: Optional[int] = None

    run_id: str
    problem_id: int

    # Energy dynamics
    mean_energy: Optional[float] = None
    max_energy: Optional[float] = None
    energy_std: Optional[float] = None

    drift_slope: Optional[float] = None
    energy_auc: Optional[float] = None

    first_drift_iteration: Optional[int] = None
    first_unstable_iteration: Optional[int] = None
    collapse_iteration: Optional[int] = None

    collapse_detected: bool = False

    # Recovery metrics
    intervention_count: int = 0

    recovery_delta: Optional[float] = None
    recovered: Optional[bool] = None

    # Accuracy outcomes
    initial_accuracy: Optional[float] = None
    final_accuracy: Optional[float] = None
    accuracy_recovered: Optional[bool] = None

    # Signal features
    max_foreign_ratio: Optional[float] = None
    mean_repetition: Optional[float] = None
