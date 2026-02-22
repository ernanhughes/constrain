from typing import Dict
from constrain.policy.custom_types import PolicyDecision, Thresholds
from constrain.reasoning_state import ReasoningState


class Policy:

    def decide(
        self,
        *,
        axes: Dict,
        metrics: Dict,
        state: ReasoningState,
        thresholds: Thresholds,
    ) -> PolicyDecision:
        raise NotImplementedError