"""
DTO for reasoning state snapshots.
"""
from typing import List, Optional
from pydantic import BaseModel, ConfigDict, Field
import json


class ReasoningStateSnapshotDTO(BaseModel):
    model_config = ConfigDict(extra="forbid", from_attributes=True)

    # Identifiers
    id: Optional[int] = None
    run_id: str
    problem_id: int
    step_id: Optional[int] = None
    
    # Existing fields
    iteration: int
    prompt_text: str
    current_reasoning: str
    history: List[str] = Field(default_factory=list)
    history_json: Optional[str] = None
    temperature: float
    initial_temperature: float
    stack_depth: int = Field(ge=1)
    is_after_revert: bool = False
    is_after_reset: bool = False
    parent_snapshot_id: Optional[int] = None
    total_energy: Optional[float] = None
    grounding_energy: Optional[float] = None
    stability_energy: Optional[float] = None
    policy_action: Optional[str] = None
    policy_action_reason: Optional[str] = None
    created_at: float
    
    # NEW FIELDS
    attempt: int = 0
    energy_slope: Optional[float] = None
    violation_level: Optional[str] = None
    consecutive_hard: Optional[int] = 0
    consecutive_rising: Optional[int] = 0
    collapse_flag: bool = False

    @classmethod
    def from_orm(cls, orm_obj) -> "ReasoningStateSnapshotDTO":
        history = []
        if orm_obj.history_json:
            try:
                history = json.loads(orm_obj.history_json)
            except (json.JSONDecodeError, TypeError):
                history = []
        
        return cls(
            id=orm_obj.id,
            run_id=orm_obj.run_id,
            problem_id=orm_obj.problem_id,
            step_id=orm_obj.step_id,
            iteration=orm_obj.iteration,
            prompt_text=orm_obj.prompt_text,
            current_reasoning=orm_obj.current_reasoning,
            history=history,
            history_json=orm_obj.history_json,
            temperature=orm_obj.temperature,
            initial_temperature=orm_obj.initial_temperature,
            stack_depth=orm_obj.stack_depth,
            is_after_revert=orm_obj.is_after_revert,
            is_after_reset=orm_obj.is_after_reset,
            parent_snapshot_id=orm_obj.parent_snapshot_id,
            total_energy=orm_obj.total_energy,
            grounding_energy=orm_obj.grounding_energy,
            stability_energy=orm_obj.stability_energy,
            policy_action=orm_obj.policy_action,
            policy_action_reason=orm_obj.policy_action_reason,
            created_at=orm_obj.created_at,
            attempt=getattr(orm_obj, "attempt", 0),
            energy_slope=getattr(orm_obj, "energy_slope", None),
            violation_level=getattr(orm_obj, "violation_level", None),
            consecutive_hard=getattr(orm_obj, "consecutive_hard", 0),
            consecutive_rising=getattr(orm_obj, "consecutive_rising", 0),
            collapse_flag=getattr(orm_obj, "collapse_flag", False),
        )

    def to_orm_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "problem_id": self.problem_id,
            "step_id": self.step_id,
            "iteration": self.iteration,
            "prompt_text": self.prompt_text,
            "current_reasoning": self.current_reasoning,
            "history_json": json.dumps(self.history),
            "temperature": self.temperature,
            "initial_temperature": self.initial_temperature,
            "stack_depth": self.stack_depth,
            "is_after_revert": self.is_after_revert,
            "is_after_reset": self.is_after_reset,
            "parent_snapshot_id": self.parent_snapshot_id,
            "total_energy": self.total_energy,
            "grounding_energy": self.grounding_energy,
            "stability_energy": self.stability_energy,
            "policy_action": self.policy_action,
            "policy_action_reason": self.policy_action_reason,
            "created_at": self.created_at,
            "attempt": self.attempt,
            "energy_slope": self.energy_slope,
            "violation_level": self.violation_level,
            "consecutive_hard": self.consecutive_hard,
            "consecutive_rising": self.consecutive_rising,
            "collapse_flag": self.collapse_flag,
        }