from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, field_validator, field_serializer


RecoveryExperimentType = Literal[
    "intervention_recovery",
    "energy_delta",
    "timing_analysis",
]

RecoveryExperimentStatus = Literal[
    "running",
    "completed",
    "failed",
]


class RecoveryExperimentDTO(BaseModel):
    """DTO for recovery experiment metadata and results."""

    model_config = ConfigDict(from_attributes=True, extra="forbid")

    # Primary key
    id: Optional[int] = None

    # Experiment metadata
    experiment_name: str
    experiment_type: RecoveryExperimentType

    # Scope configuration (stored as JSON strings in DB, lists/dicts in DTO)
    run_ids: List[str]
    problem_filter: Optional[Dict[str, Any]] = None

    # Analysis configuration
    energy_threshold: Optional[float] = None
    accuracy_delta_threshold: Optional[float] = None
    min_pre_intervention_steps: Optional[int] = None
    min_post_intervention_steps: Optional[int] = None

    # Timing
    start_time: float
    end_time: Optional[float] = None
    status: RecoveryExperimentStatus = "running"

    # Results (stored as JSON strings in DB, dicts in DTO)
    summary_metrics: Optional[Dict[str, Any]] = None
    per_problem_results: Optional[Dict[str, Any]] = None
    statistical_tests: Optional[Dict[str, Any]] = None

    # Diagnostics
    overlap_warning: bool = False
    confounding_warning: bool = False
    notes: Optional[str] = None

    # ============================================================
    # Validators: Parse JSON strings from ORM → native Python types
    # ============================================================

    @field_validator("run_ids", mode="before")
    @classmethod
    def parse_run_ids(cls, v):
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

    @field_validator(
        "problem_filter",
        "summary_metrics",
        "per_problem_results",
        "statistical_tests",
        mode="before",
    )
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
    # Serializers: Convert native types → JSON strings for ORM
    # ============================================================

    @field_serializer("run_ids")
    def serialize_run_ids(self, v: List[str]) -> Optional[str]:
        return json.dumps(v) if v is not None else None

    @field_serializer(
        "problem_filter",
        "summary_metrics",
        "per_problem_results",
        "statistical_tests",
    )
    def serialize_json_dict(self, v: Optional[Dict[str, Any]]) -> Optional[str]:
        return json.dumps(v) if v is not None else None

    # ============================================================
    # Factory & helper methods
    # ============================================================

    @classmethod
    def create(
        cls,
        experiment_name: str,
        experiment_type: RecoveryExperimentType,
        run_ids: List[str],
        problem_filter: Optional[Dict[str, Any]] = None,
        energy_threshold: Optional[float] = None,
        accuracy_delta_threshold: Optional[float] = None,
        min_pre_intervention_steps: Optional[int] = None,
        min_post_intervention_steps: Optional[int] = None,
        notes: Optional[str] = None,
    ) -> "RecoveryExperimentDTO":
        """Factory method to create a new recovery experiment DTO."""
        return cls(
            experiment_name=experiment_name,
            experiment_type=experiment_type,
            run_ids=run_ids,
            problem_filter=problem_filter,
            energy_threshold=energy_threshold,
            accuracy_delta_threshold=accuracy_delta_threshold,
            min_pre_intervention_steps=min_pre_intervention_steps,
            min_post_intervention_steps=min_post_intervention_steps,
            start_time=time.time(),
            status="running",
            overlap_warning=False,
            confounding_warning=False,
            notes=notes,
        )

    def add_results(
        self,
        summary_metrics: Dict[str, Any],
        statistical_tests: Optional[Dict[str, Any]] = None,
        per_problem_results: Optional[Dict[str, Any]] = None,
    ) -> "RecoveryExperimentDTO":
        """Add analysis results to the experiment."""
        return self.model_copy(
            update={
                "summary_metrics": summary_metrics,
                "statistical_tests": statistical_tests,
                "per_problem_results": per_problem_results,
            }
        )

    def mark_completed(self) -> "RecoveryExperimentDTO":
        """Mark experiment as completed and set end_time."""
        return self.model_copy(
            update={"status": "completed", "end_time": time.time()}
        )

    def mark_failed(self, error_notes: Optional[str] = None) -> "RecoveryExperimentDTO":
        """Mark experiment as failed."""
        notes = f"{self.notes or ''}\nError: {error_notes}".strip()
        return self.model_copy(
            update={"status": "failed", "end_time": time.time(), "notes": notes}
        )

    def set_diagnostics(
        self,
        overlap_warning: bool = False,
        confounding_warning: bool = False,
    ) -> "RecoveryExperimentDTO":
        """Set diagnostic warnings for causal validity."""
        return self.model_copy(
            update={
                "overlap_warning": overlap_warning,
                "confounding_warning": confounding_warning,
            }
        )