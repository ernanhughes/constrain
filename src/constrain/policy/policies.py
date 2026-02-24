# constrain/policy/policies.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from constrain.reasoning_state import ReasoningState

from .custom_types import Policy, PolicyDecision, Thresholds
from .policy_params import PolicyParams
from .soft_intervention_engine import SoftInterventionEngine, RandomizedSoftInterventionPolicy
import numpy as np

import logging 
logger = logging.getLogger(__name__)

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
    policy_id: int = 7
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
    policy_id: int = 8
    name: str = "learned"

    def decide(self, *, axes: Axes, metrics: Metrics, thresholds: Thresholds, state: ReasoningState) -> PolicyDecision:    
        action, collapse_prob, value = self.learned_model.decide(metrics)
        
        # ── NORMALIZE collapse_prob to float ──────────────────────────────
        # Handle dict, None, or scalar outputs from learned model
        if isinstance(collapse_prob, dict):
            # Prefer 'p_accept' as primary signal, fallback to first value, then 0.0
            collapse_prob = float(collapse_prob.get("p_accept", next(iter(collapse_prob.values()), 0.0)))
        elif collapse_prob is None:
            collapse_prob = 0.0
        else:
            collapse_prob = float(collapse_prob)  # Ensure it's a Python float, not np.float32 etc.
        # ──────────────────────────────────────────────────────────────────
        
        logger.debug(f"[LearnedPolicyWrapper] action={action} collapse_prob={collapse_prob} value={value}")

        # shadow = observe but do not intervene
        if self.shadow:
            return PolicyDecision(
                action="ACCEPT",
                new_temperature=state.temperature,
                collapse_probability=collapse_prob,  # ✅ Now guaranteed float
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
            meta={"learned_value": value},
        )


@dataclass
class SoftInterventionPolicyWrapper(Policy):
    """
    Wraps SoftInterventionEngine for use in PolicyEngine.
    Supports both deterministic and randomized modes.
    """
    params: PolicyParams
    engine: SoftInterventionEngine
    shadow: bool = False
    randomized: bool = False
    randomization_rate: float = 0.5
    policy_id: int = 9  # or 101 for randomized
    name: str = "soft_intervention"

    def decide(
        self,
        *,
        axes: Axes,
        metrics: Metrics,
        state: ReasoningState,
        thresholds: Thresholds,
    ) -> PolicyDecision:
        # Extract risk score (from learned model or energy-based heuristic)
        risk_score = float(metrics.get("collapse_probability", axes.get("energy", 0.0)))
        
        # Get soft intervention decision
        decision = self.engine.decide(
            risk_score=risk_score,
            current_temperature=state.temperature,
            min_temperature=self.params.min_temperature,
        )
        
        # Handle randomized mode
        if self.randomized and risk_score > thresholds.tau_medium:
            import random
            if random.random() < self.randomization_rate:
                # Override to ACCEPT for causal comparison
                final_action = "ACCEPT"
                randomized_flag = True
                propensity = 1 - self.randomization_rate
            else:
                final_action = decision.mode  # Use recommended mode
                randomized_flag = True
                propensity = self.randomization_rate
        else:
            final_action = decision.mode
            randomized_flag = False
            propensity = 1.0 if decision.mode != "ACCEPT" else 0.0

        # Map soft mode to legacy action format
        action = "ACCEPT"
        if final_action in ("NUDGE", "STABILIZE", "CORRECT"):
            action = "REVERT"
        elif final_action == "RESET":
            action = "RESET"

        # Compute new temperature
        new_temperature = decision.new_temperature

        # Build intervention metadata for logging
        intervention_meta = {
            "recommended_mode": decision.mode,
            "final_mode": final_action,
            "intensity": decision.intensity,
            "risk_score": risk_score,
            "randomized": randomized_flag,
            "propensity_score": propensity,
            "prompt_modification": decision.prompt_modification,
        }

        # Shadow mode: observe but don't intervene
        if self.shadow:
            action = "ACCEPT"
            new_temperature = state.temperature
            intervention_meta["shadow_mode"] = True

        return PolicyDecision(
            action=action,
            new_temperature=new_temperature,
            collapse_probability=risk_score,
            meta=intervention_meta,
        )


@dataclass
class RandomizedSoftInterventionPolicyWrapper(Policy):
    """
    Policy adapter for RandomizedSoftInterventionPolicy.
    
    Bridges the causal randomization engine to the Policy interface.
    Policy ID: 10 (matches your experiment config)
    """
    params: PolicyParams
    engine: SoftInterventionEngine
    risk_threshold: float = 0.65
    randomization_rate: float = 0.5
    seed: Optional[int] = None
    shadow: bool = False
    policy_id: int = 10
    name: str = "randomized_soft_intervention"

    def decide(
        self,
        *,
        axes: Axes,
        metrics: Metrics,
        state: ReasoningState,
        thresholds: Thresholds,
    ) -> PolicyDecision:
        # ── Extract risk score (with fallback) ──────────────────────────
        risk_score = float(
            metrics.get("collapse_probability") 
            or axes.get("energy", 0.0)
        )
        risk_score = np.clip(risk_score, 0.0, 1.0)  # Ensure valid probability
        
        # ── We need problem_id/iteration for deterministic randomization ─
        # These come from the runner context; pass via metrics or state if needed
        problem_id = metrics.get("problem_id", "unknown")
        iteration = metrics.get("iteration", 0)
        run_id = metrics.get("run_id", "unknown")
        
        # ── Delegate to randomized engine ───────────────────────────────
        randomized_policy = RandomizedSoftInterventionPolicy(
            engine=self.engine,
            risk_threshold=self.risk_threshold,
            randomization_rate=self.randomization_rate,
            seed=self.seed,
        )
        
        decision, meta = randomized_policy.decide(
            risk_score=risk_score,
            current_temperature=state.temperature,
            problem_id=str(problem_id),
            iteration=iteration,
            run_id=str(run_id),
            min_temperature=self.params.min_temperature,
        )
        
        # ── Map SoftInterventionDecision → PolicyDecision ───────────────
        action = self._map_mode_to_action(decision.mode)
        
        new_temperature = decision.new_temperature
        if action == "REVERT" and decision.mode != "RESET":
            # Apply cooldown factor for reverts (consistent with other policies)
            new_temperature = max(
                self.params.min_temperature, 
                state.temperature * self.params.revert_cooldown_factor
            )
        
        # ── Build rich meta for forensic auditing ───────────────────────
        policy_meta = {
            "soft_mode": decision.mode,
            "intensity": decision.intensity,
            "risk_score": risk_score,
            "decision_type": meta["decision_type"],  # "randomized_accept" etc.
            "propensity_score": self._compute_propensity(meta),
            "prompt_modification": decision.prompt_modification,
            "shadow": self.shadow,
        }
        
        # Shadow mode: observe but don't intervene
        if self.shadow:
            action = "ACCEPT"
            new_temperature = state.temperature
            policy_meta["shadow_mode"] = True
        
        return PolicyDecision(
            action=action,
            new_temperature=new_temperature,
            collapse_probability=risk_score,  # ✅ Float, as required
            meta=policy_meta,
        )
    
    def _map_mode_to_action(self, mode: str) -> str:
        """Map soft intervention modes to legacy action format."""
        if mode == "ACCEPT":
            return "ACCEPT"
        elif mode == "RESET":
            return "RESET"
        else:  # NUDGE, STABILIZE, CORRECT → REVERT
            return "REVERT"
    
    def _compute_propensity(self, meta: Dict) -> float:
        """Compute propensity score for causal inference."""
        decision_type = meta.get("decision_type", "deterministic")
        if decision_type == "randomized_accept":
            return 1.0 - self.randomization_rate  # P(assigned to ACCEPT)
        elif decision_type == "randomized_intervene":
            return self.randomization_rate  # P(assigned to intervene)
        return 1.0  # Deterministic decisions have propensity=1