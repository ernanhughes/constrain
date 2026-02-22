from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol, Optional, Dict, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from constrain.data.memory import Memory
from constrain.reasoning_state import ReasoningState


@dataclass(frozen=True)
class Thresholds:
    tau_soft: float
    tau_medium: float
    tau_hard: float


@dataclass
class PolicyDecision:
    action: str
    new_temperature: float
    collapse_probability: Optional[float] = None
    meta: Optional[Dict[str, Any]] = None


class ThresholdProvider(Protocol):
    def get(
        self, *, cfg, memory: Memory, run_id: str, step_id: int | None = None
    ) -> Thresholds: ...


class Policy(Protocol):
    policy_id: int
    name: str

    def decide(
        self,
        *,
        axes: Dict[str, float],
        metrics: Dict[str, float],
        state: ReasoningState,
        thresholds: Thresholds,
    ) -> PolicyDecision: ...
