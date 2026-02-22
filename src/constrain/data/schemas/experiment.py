from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, field_validator, field_serializer


class ExperimentDTO(BaseModel):
    """Experiment metadata for grouping related runs."""
    
    model_config = ConfigDict(from_attributes=True, extra="forbid")

    id: Optional[int] = None

    # Experiment metadata
    experiment_name: str
    experiment_type: str  # "policy_comparison" | "threshold_sweep" | "ablation"

    # Configuration (stored as JSON strings in DB, lists in DTO)
    policy_ids: List[int] = []
    seeds: List[int] = []
    num_problems: Optional[int] = None
    num_recursions: Optional[int] = None

    # Timing
    start_time: float
    end_time: Optional[float] = None
    status: str = "running"  # "running" | "completed" | "failed"

    # Results summary
    results_summary: Optional[Dict[str, Any]] = None

    # Notes / metadata
    notes: Optional[str] = None
    git_commit: Optional[str] = None

    # ============================================================
    # Validators: Parse JSON strings from ORM
    # ============================================================

    @field_validator("policy_ids", "seeds", mode="before")
    @classmethod
    def parse_json_list(cls, v):
        if v is None:
            return []
        if isinstance(v, str):
            try:
                return json.loads(v)
            except (json.JSONDecodeError, TypeError):
                return []
        if isinstance(v, (list, tuple)):
            return list(v)
        return []

    @field_validator("results_summary", mode="before")
    @classmethod
    def parse_json_dict(cls, v):
        if v is None:
            return None
        if isinstance(v, str):
            try:
                return json.loads(v)
            except (json.JSONDecodeError, TypeError):
                return None
        if isinstance(v, dict):
            return v
        return None

    # ============================================================
    # Serializers: Convert lists/dicts to JSON strings for ORM
    # ============================================================

    @field_serializer("policy_ids", "seeds")
    def serialize_json_list(self, v):
        return json.dumps(v) if v is not None else None

    @field_serializer("results_summary")
    def serialize_json_dict(self, v):
        return json.dumps(v) if v is not None else None

    # ============================================================
    # Helpers
    # ============================================================

    @classmethod
    def create(
        cls,
        experiment_name: str,
        experiment_type: str,
        policy_ids: List[int],
        seeds: List[int],
        num_problems: Optional[int] = None,
        num_recursions: Optional[int] = None,
        notes: Optional[str] = None,
    ) -> "ExperimentDTO":
        """Factory method to create a new experiment DTO."""
        import time
        return cls(
            experiment_name=experiment_name,
            experiment_type=experiment_type,
            policy_ids=policy_ids,
            seeds=seeds,
            num_problems=num_problems,
            num_recursions=num_recursions,
            start_time=time.time(),
            status="running",
            notes=notes,
        )