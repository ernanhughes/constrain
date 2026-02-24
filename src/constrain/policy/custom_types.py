from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, Optional, Protocol

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

    decision_mode: str = "deterministic"
    propensity: float = 1.0


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
