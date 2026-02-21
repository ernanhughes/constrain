# constrain/data/schemas/experiment.py

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class ExperimentDTO(BaseModel):
    model_config = ConfigDict(extra="forbid", from_attributes=True)

    id: Optional[int] = None

    # Experiment metadata
    experiment_name: str
    experiment_type: str  # "policy_comparison" | "threshold_sweep" | "ablation"

    # Configuration
    policy_ids: List[int] = Field(default_factory=list)  # Will be JSON-encoded/decoded
    seeds: List[int] = Field(default_factory=list)
    num_problems: Optional[int] = None
    num_recursions: Optional[int] = None

    # Timing
    start_time: float
    end_time: Optional[float] = None
    status: str  # "running" | "completed" | "failed"

    # Results summary
    results_summary: Optional[Dict[str, Any]] = None

    # Notes / metadata
    notes: Optional[str] = None
    git_commit: Optional[str] = None

    # Helper to parse JSON fields
    @classmethod
    def from_orm(cls, orm_obj) -> "ExperimentDTO":
        import json

        data = {
            "id": orm_obj.id,
            "experiment_name": orm_obj.experiment_name,
            "experiment_type": orm_obj.experiment_type,
            "policy_ids": json.loads(orm_obj.policy_ids) if orm_obj.policy_ids else [],
            "seeds": json.loads(orm_obj.seeds) if orm_obj.seeds else [],
            "num_problems": orm_obj.num_problems,
            "num_recursions": orm_obj.num_recursions,
            "start_time": orm_obj.start_time,
            "end_time": orm_obj.end_time,
            "status": orm_obj.status,
            "results_summary": json.loads(orm_obj.results_summary) if orm_obj.results_summary else None,
            "notes": orm_obj.notes,
            "git_commit": orm_obj.git_commit,
        }
        return cls(**data)