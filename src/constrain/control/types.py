"""
Control system types - shared definitions, no logic.
"""
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class ViolationLevel(str, Enum):
    """Violation severity levels (global, policy-independent)."""
    NONE = "none"
    SOFT = "soft"
    MEDIUM = "medium"
    HARD = "hard"


class Action(str, Enum):
    """Policy enforcement actions."""
    ALLOW = "allow"
    ROLLBACK = "rollback"
    RESET = "reset"
    TERMINATE = "terminate"


@dataclass(frozen=True)
class EnergyDynamics:
    """
    Energy dynamics state at a single step.
    
    Passed to policy for decision-making.
    Persisted to database for analysis.
    """
    energy: float
    prev_energy: Optional[float]
    slope: float
    violation: ViolationLevel
    consecutive_hard: int
    consecutive_rising: int
    collapse_flag: bool
    attempt: int  # Model call count (increments every loop)
    iteration: int  # Accepted depth (increments only on ALLOW)


@dataclass(frozen=True)
class ControlDecision:
    """
    Policy enforcement decision.
    
    Deterministic output from policy.evaluate()
    """
    action: Action
    new_temperature: float
    reason: str