# constrain/policy/exploration_wrapper.py

import numpy as np
from dataclasses import dataclass
from .custom_types import Policy, PolicyDecision

@dataclass
class ExplorationWrapper:
    base_policy: Policy
    epsilon: float = 0.2  # exploration rate
    rng_seed: int = 42

    def __post_init__(self):
        self.rng = np.random.RandomState(self.rng_seed)

    def decide(self, axes, metrics, state, thresholds) -> PolicyDecision:

        # Base deterministic decision
        base_decision = self.base_policy.decide(
            axes=axes,
            metrics=metrics,
            state=state,
            thresholds=thresholds,
        )

        # Exploration flip
        if self.rng.rand() < self.epsilon:
            flipped_action = (
                "ACCEPT" if base_decision.action != "ACCEPT" else "REVERT"
            )

            return PolicyDecision(
                action=flipped_action,
                new_temperature=base_decision.new_temperature,
                collapse_probability=base_decision.collapse_probability,
                decision_mode="exploration",
                propensity=self.epsilon,
            )

        return PolicyDecision(
            action=base_decision.action,
            new_temperature=base_decision.new_temperature,
            collapse_probability=base_decision.collapse_probability,
            decision_mode="deterministic",
            propensity=1 - self.epsilon,
        )