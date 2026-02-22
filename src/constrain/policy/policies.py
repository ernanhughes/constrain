# constrain/policy/policies.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Protocol

from constrain.reasoning_state import ReasoningState
from .custom_types import Policy, PolicyDecision, Thresholds
from .policy_params import PolicyParams


Axes = Dict[str, Any]
Metrics = Dict[str, Any]


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def _energy(axes: Axes) -> float:
    return float(axes.get("energy", 0.0))


# ------------------------------------------------------------
# Policies
# ------------------------------------------------------------

@dataclass
class BaselineAcceptPolicy(Policy):
    params: PolicyParams
    policy_id: int = 0
    name: str = "accept_all"

    def decide(self, *, axes: Axes, metrics: Metrics, state: ReasoningState, thresholds: Thresholds) -> PolicyDecision:
        return PolicyDecision(action="ACCEPT", new_temperature=state.temperature)


@dataclass
class SimpleRevertPolicy(Policy):
    params: PolicyParams
    policy_id: int = 1
    name: str = "simple_revert"

    def decide(self, *, axes: Axes, metrics: Metrics, state: ReasoningState, thresholds: Thresholds) -> PolicyDecision:
        return PolicyDecision(
            action="REVERT",
            new_temperature=max(self.params.min_temperature, state.temperature * self.params.revert_cooldown_factor),
        )


@dataclass
class RevertCoolPolicy(Policy):
    params: PolicyParams
    policy_id: int = 2
    name: str = "revert_cool"

    def decide(self, *, axes: Axes, metrics: Metrics, state: ReasoningState, thresholds: Thresholds) -> PolicyDecision:
        # Same as SimpleRevert right now; you can differentiate later via params
        return PolicyDecision(
            action="REVERT",
            new_temperature=max(self.params.min_temperature, state.temperature * self.params.revert_cooldown_factor),
        )


@dataclass
class AggressiveRevertPolicy(Policy):
    params: PolicyParams
    policy_id: int = 3
    name: str = "aggressive_revert"

    def decide(self, *, axes: Axes, metrics: Metrics, state: ReasoningState, thresholds: Thresholds) -> PolicyDecision:
        e = _energy(axes)
        t = state.temperature

        if e <= thresholds.tau_soft:
            return PolicyDecision(action="ACCEPT", new_temperature=t, meta={"energy": e})

        # REVERT path
        if e > thresholds.tau_medium:
            new_t = max(self.params.min_temperature, t * self.params.aggressive_cooldown_factor)
        else:
            new_t = max(self.params.min_temperature, t * self.params.revert_cooldown_factor)

        return PolicyDecision(action="REVERT", new_temperature=new_t, meta={"energy": e})


@dataclass
class HardResetPolicy(Policy):
    params: PolicyParams
    policy_id: int = 4
    name: str = "hard_reset"

    def decide(self, *, axes: Axes, metrics: Metrics, state: ReasoningState, thresholds: Thresholds) -> PolicyDecision:
        e = _energy(axes)
        t = state.temperature

        if e > thresholds.tau_hard:
            return PolicyDecision(
                action="RESET",
                new_temperature=max(self.params.min_temperature, t * self.params.reset_cooldown_factor),
                meta={"energy": e},
            )

        return PolicyDecision(
            action="REVERT",
            new_temperature=max(self.params.min_temperature, t * self.params.revert_cooldown_factor),
            meta={"energy": e},
        )


@dataclass
class MediumResetPolicy(Policy):
    params: PolicyParams
    policy_id: int = 5
    name: str = "medium_reset"

    def decide(self, *, axes: Axes, metrics: Metrics, state: ReasoningState, thresholds: Thresholds) -> PolicyDecision:
        e = _energy(axes)
        t = state.temperature

        action = "RESET" if e > thresholds.tau_medium else "REVERT"

        return PolicyDecision(
            action=action,
            new_temperature=max(self.params.min_temperature, t * self.params.revert_cooldown_factor),
            meta={"energy": e},
        )


@dataclass
class RandomPolicy(Policy):
    params: PolicyParams
    revert_probability: float
    policy_id: int = 6
    name: str = "random"

    def decide(self, *, axes: Axes, metrics: Metrics, state: ReasoningState, thresholds: Thresholds) -> PolicyDecision:
        import random

        if random.random() < float(self.revert_probability):
            return PolicyDecision(
                action="REVERT",
                new_temperature=max(self.params.min_temperature, state.temperature * self.params.revert_cooldown_factor),
            )

        return PolicyDecision(action="ACCEPT", new_temperature=state.temperature)


@dataclass
class GeometryBandPolicy(Policy):
    params: PolicyParams
    pr_threshold: float
    sensitivity_threshold: float
    band_width_factor: float
    policy_id: int = 8
    name: str = "geometry_band"

    def decide(self, *, axes: Axes, metrics: Metrics, state: ReasoningState, thresholds: Thresholds) -> PolicyDecision:
        e = _energy(axes)
        pr = float(axes.get("participation_ratio", 0.0))
        sens = float(axes.get("sensitivity", 0.0))

        gap = thresholds.tau_soft * float(self.band_width_factor)
        low = thresholds.tau_soft - gap
        high = thresholds.tau_soft + gap

        if e >= high:
            action = "REVERT"
        elif e > low and (pr > float(self.pr_threshold) or sens > float(self.sensitivity_threshold)):
            action = "REVERT"
        else:
            action = "ACCEPT"

        # NOTE: if you want cooling on REVERT here, do it via params (like others)
        new_t = state.temperature
        if action == "REVERT":
            new_t = max(self.params.min_temperature, state.temperature * self.params.revert_cooldown_factor)

        return PolicyDecision(action=action, new_temperature=new_t, meta={"energy": e, "pr": pr, "sens": sens})


@dataclass
class LearnedPolicyWrapper(Policy):
    """
    Wraps your LearnedPolicy model, but returns PolicyDecision and uses params for temperature logic.
    """
    params: PolicyParams
    learned_model: Any
    shadow: bool = False
    policy_id: int = 99
    name: str = "learned"

    def decide(self, *, axes: Axes, metrics: Metrics, thresholds: Thresholds, state: ReasoningState) -> PolicyDecision:
        action, collapse_prob = self.learned_model.decide(metrics)

        # shadow = observe but do not intervene
        if self.shadow:
            return PolicyDecision(
                action="ACCEPT",
                new_temperature=state.temperature,
                collapse_probability=collapse_prob,
                meta={"shadow_action": action},
            )

        new_temperature = state.temperature
        if action == "REVERT":
            new_temperature = max(self.params.min_temperature, state.temperature * self.params.revert_cooldown_factor)
        elif action == "RESET":
            new_temperature = max(self.params.min_temperature, state.temperature * self.params.reset_cooldown_factor)

        return PolicyDecision(
            action=action,
            new_temperature=new_temperature,
            collapse_probability=collapse_prob,
        )