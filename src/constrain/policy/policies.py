from __future__ import annotations
from dataclasses import dataclass

from .custom_types import Policy, PolicyDecision, Thresholds
from constrain.reasoning_state import ReasoningState
from .policy_params import PolicyParams


@dataclass
class BaselineAcceptPolicy:
    policy_id: int = 0
    name: str = "accept_all"

    def decide(
        self, *, axes, metrics, state: ReasoningState, thresholds: Thresholds
    ) -> PolicyDecision:
        return PolicyDecision(
            action="ACCEPT",
            new_temperature=state.temperature,
        )


@dataclass
class AggressiveRevertPolicy:
    params: PolicyParams
    policy_id: int = 3
    name: str = "aggressive_revert"

    def decide(
        self, *, axes, metrics, state: ReasoningState, thresholds: Thresholds
    ) -> PolicyDecision:
        e = float(axes.get("energy", 0.0))
        t = state.temperature

        action = "ACCEPT"
        new_t = t

        if e > thresholds.tau_soft:
            action = "REVERT"

            if e > thresholds.tau_medium:
                new_t = max(
                    self.params.min_temperature,
                    t * self.params.aggressive_cooldown_factor,
                )
            else:
                new_t = max(
                    self.params.min_temperature,
                    t * self.params.revert_cooldown_factor,
                )

        return PolicyDecision(
            action=action,
            new_temperature=new_t,
            meta={"energy": e},
        )


@dataclass
class HardResetPolicy:
    params: PolicyParams
    policy_id: int = 4
    name: str = "hard_reset"

    def decide(
        self, *, axes, metrics, state: ReasoningState, thresholds: Thresholds
    ) -> PolicyDecision:
        e = float(axes.get("energy", 0.0))
        t = state.temperature

        if e > thresholds.tau_hard:
            return PolicyDecision(
                action="RESET",
                new_temperature=max(
                    self.params.min_temperature,
                    t * self.params.reset_cooldown_factor,
                ),
                meta={"energy": e},
            )

        return PolicyDecision(
            action="REVERT",
            new_temperature=max(
                self.params.min_temperature,
                t * self.params.revert_cooldown_factor,
            ),
            meta={"energy": e},
        )


@dataclass
class SimpleRevertPolicy:
    params: PolicyParams
    policy_id: int = 1
    name: str = "simple_revert"

    def decide(self, *, axes, metrics, state: ReasoningState, thresholds: Thresholds) -> PolicyDecision:
        return PolicyDecision(
            action="REVERT",
            new_temperature=max(
                self.params.min_temperature,
                state.temperature * self.params.revert_cooldown_factor,
            ),
        )


@dataclass
class RevertCoolPolicy:
    params: PolicyParams
    policy_id: int = 2
    name: str = "revert_cool"

    def decide(
        self, *, axes, metrics, state: ReasoningState, thresholds: Thresholds
    ) -> PolicyDecision:
        return PolicyDecision(
            action="REVERT",
            new_temperature=max(
                self.params.min_temperature,
                state.temperature * self.params.revert_cooldown_factor,
            ),
        )


@dataclass
class MediumResetPolicy:
    params: PolicyParams
    policy_id: int = 5
    name: str = "medium_reset"

    def decide(
        self, *, axes, metrics, state: ReasoningState, thresholds: Thresholds
    ) -> PolicyDecision:
        e = float(axes.get("energy", 0.0))
        t = state.temperature

        if e > thresholds.tau_medium:
            action = "RESET"
        else:
            action = "REVERT"

        return PolicyDecision(
            action=action,
            new_temperature=max(
                self.params.min_temperature,
                t * self.params.revert_cooldown_factor,
            ),
            meta={"energy": e},
        )

@dataclass
class RandomPolicy:
    params: PolicyParams
    revert_probability: float
    policy_id: int = 6
    name: str = "random"

    def decide(self, *, axes, metrics, state: ReasoningState, thresholds: Thresholds) -> PolicyDecision:

        import random

        if random.random() < self.revert_probability:
            return PolicyDecision(
                action="REVERT",
                new_temperature=max(
                    self.params.min_temperature,
                    state.temperature * self.params.revert_cooldown_factor,
                ),
            )

        return PolicyDecision(
            action="ACCEPT",
            new_temperature=state.temperature,
        )
    

@dataclass
class GeometryBandPolicy:
    params: PolicyParams
    pr_threshold: float
    sensitivity_threshold: float
    band_width_factor: float
    policy_id: int = 8
    name: str = "geometry_band"

    def decide(self, *, axes, metrics, state: ReasoningState, thresholds: Thresholds) -> PolicyDecision:

        e = float(axes.get("energy", 0.0))
        pr = float(axes.get("participation_ratio", 0.0))
        sens = float(axes.get("sensitivity", 0.0))

        gap = thresholds.tau_soft * self.band_width_factor
        low = thresholds.tau_soft - gap
        high = thresholds.tau_soft + gap

        if e >= high:
            action = "REVERT"
        elif e > low and (pr > self.pr_threshold or sens > self.sensitivity_threshold):
            action = "REVERT"
        else:
            action = "ACCEPT"

        return PolicyDecision(
            action=action,
            new_temperature=state.temperature,
            meta={"energy": e, "pr": pr, "sens": sens},
        )
    
class LearnedPolicyWrapper(Policy):

    def __init__(self, learned_model, shadow=False):
        self.model = learned_model
        self.shadow = shadow

    def decide(self, *, axes, flat_metrics, thresholds, state):

        action, collapse_prob = self.model.decide(flat_metrics)

        if self.shadow:
            return "ACCEPT", state.temperature, collapse_prob

        new_temperature = state.temperature

        if action == "REVERT":
            new_temperature = max(0.1, state.temperature * 0.9)

        return action, new_temperature, collapse_prob