# constrain/policy/apply_policy.py
"""
Policy Application Layer

Integrates soft intervention engine with existing policy registry.
"""

from __future__ import annotations
import os
import random
from typing import Dict, Optional, Tuple
from constrain.calibration.recursive import RecursiveCalibrator
from constrain.config import get_config
from constrain.data.memory import Memory
from constrain.policy.learned_policy import LearnedPolicy
from .soft_intervention_engine import (
    SoftInterventionEngine,
    RandomizedSoftInterventionPolicy,
)
from constrain.reasoning_state import ReasoningState

_learned_policy_instance = None
_soft_intervention_engine = None


# ============================================================
# Threshold Selection
# ============================================================

def _get_thresholds(cfg, memory: Memory, run_id: str):
    if cfg.policy_mode == "static":
        return cfg.tau_soft, cfg.tau_medium, cfg.tau_hard

    if cfg.policy_mode == "recursive":
        return RecursiveCalibrator.get_thresholds(memory, run_id)

    raise ValueError(f"Unknown policy_mode: {cfg.policy_mode}")


# ============================================================
# Learned Policy Loader (Singleton)
# ============================================================

def _get_learned_policy(threshold: float = None) -> Optional[LearnedPolicy]:
    global _learned_policy_instance
    cfg = get_config()

    if not hasattr(cfg, "learned_model_path"):
        print("No learned_model_path in config, skipping learned policy.")
        return None

    if not os.path.exists(cfg.learned_model_path):
        print(f"Learned model not found at {cfg.learned_model_path}, skipping learned policy.")
        return None

    if _learned_policy_instance is None:
        _learned_policy_instance = LearnedPolicy(
            model_path=cfg.learned_model_path,
            threshold=threshold if threshold is not None else cfg.learned_policy_threshold,
        )
    else:
        if threshold is not None:
            _learned_policy_instance.set_threshold(threshold)

    return _learned_policy_instance


# ============================================================
# Soft Intervention Engine Loader (Singleton)
# ============================================================

def _get_soft_intervention_engine() -> Optional[SoftInterventionEngine]:
    global _soft_intervention_engine
    
    if _soft_intervention_engine is None:
        _soft_intervention_engine = SoftInterventionEngine()
    
    return _soft_intervention_engine


# ============================================================
# Main Policy Dispatcher
# ============================================================

def apply_policy(
    policy_id: int,
    axes: Dict,
    flat_metrics: Dict,
    reasoning: str,
    state: ReasoningState,
    memory: Memory,
    run_id: str,
    threshold: float = None,
    problem_id: str = None,
    iteration: int = None,
) -> Tuple[str, float, Optional[float], Optional[Dict]]:
    """
    Apply policy decision with optional soft intervention.
    
    Returns:
        action: ACCEPT, REVERT, RESET
        new_temperature: Updated temperature
        collapse_prob: From learned model (if applicable)
        intervention_meta For causal logging
    """
    cfg = get_config()

    # Default outputs
    action = "ACCEPT"
    new_temperature = state.temperature
    collapse_prob: Optional[float] = None
    intervention_meta: Optional[Dict] = None

    energy = axes.get("energy")
    pr = axes.get("participation_ratio")
    sensitivity = axes.get("sensitivity")

    tau_soft = cfg.tau_soft
    tau_medium = cfg.tau_medium
    tau_hard = cfg.tau_hard

    # ============================================================
    # Policy 0 — Accept All (Baseline)
    # ============================================================
    if policy_id == 0:
        pass

    # ============================================================
    # Stable Region Shortcut
    # ============================================================
    elif energy <= tau_soft:
        pass

    # ============================================================
    # Policies 1-8 — Legacy Hard Interventions
    # ============================================================
    elif policy_id == 1:
        action = "REVERT"

    elif policy_id == 2:
        action = "REVERT"
        new_temperature = max(0.1, state.temperature * 0.9)

    elif policy_id == 3:
        action = "REVERT"
        if energy > tau_medium:
            new_temperature = max(0.1, state.temperature * 0.75)

    elif policy_id == 4:
        if energy > tau_hard:
            action = "RESET"
            new_temperature = max(0.1, state.temperature * 0.7)
        else:
            action = "REVERT"

    elif policy_id == 5:
        if energy > tau_medium:
            action = "RESET"
        else:
            action = "REVERT"
        new_temperature = max(0.1, state.temperature * 0.85)

    elif policy_id == 6:
        if random.random() < 0.3:
            action = "REVERT"

    elif policy_id == 7:
        r = random.random()
        if r < 0.4:
            action = "REVERT"
        elif r < 0.7:
            action = "REVERT"
            new_temperature = max(0.1, state.temperature * 0.85)
        elif r < 0.9:
            action = "RESET"
            new_temperature = max(0.1, state.temperature * 0.85)
        else:
            action = "RESET"
            new_temperature = max(0.1, state.temperature * 0.7)

    elif policy_id == 8:
        gap_width = 0.15 * tau_soft
        low = tau_soft - gap_width
        high = tau_soft + gap_width

        if energy >= high:
            action = "REVERT"
        elif energy > low:
            if pr > 0.6 or sensitivity > 0.5:
                action = "REVERT"

    # ============================================================
    # Policy 99 — Learned Policy (S-Learner Outcome Model)
    # ============================================================
    elif policy_id == 99:
        learned_policy = _get_learned_policy(threshold=threshold)
        soft_engine = _get_soft_intervention_engine()

        if learned_policy is not None and soft_engine is not None:
            # Get collapse probability from outcome model
            feature_dict = {
                "iteration": float(iteration or 0),
                "temperature": float(state.temperature),
                "ascii_ratio": float(flat_metrics.get("ascii_ratio", 1.0)),
                "foreign_char_ratio": float(flat_metrics.get("foreign_char_ratio", 0.0)),
                "repetition_score": float(flat_metrics.get("repetition_score", 0.0)),
                "total_energy": float(flat_metrics.get("total_energy", 0.0)),
                "grounding_energy": float(flat_metrics.get("grounding_energy", 0.0)),
                "stability_energy": float(flat_metrics.get("stability_energy", 0.0)),
                "is_intervention": 0,  # Counterfactual baseline
            }

            # Get risk score
            _, collapse_prob = learned_policy.decide(feature_dict)

            # Check if we should use randomized exploration
            use_randomized = getattr(cfg, "learned_policy_randomized_exploration", False)

            if use_randomized and problem_id is not None and iteration is not None:
                # Use randomized policy for causal data collection
                random_policy = RandomizedSoftInterventionPolicy(
                    engine=soft_engine,
                    risk_threshold=getattr(cfg, "learned_policy_threshold", 0.65),
                    randomization_rate=0.5,
                )

                decision, intervention_metadata = random_policy.decide(
                    risk_score=collapse_prob,
                    current_temperature=state.temperature,
                    problem_id=str(problem_id),
                    iteration=iteration,
                    run_id=run_id,
                    min_temperature=cfg.min_temperature,
                )

                # Map soft decision to legacy action format
                if decision.should_reset:
                    action = "RESET"
                elif decision.should_revert:
                    action = "REVERT"
                else:
                    action = "ACCEPT"

                new_temperature = decision.new_temperature

            else:
                # Use deterministic soft intervention
                decision = soft_engine.decide(
                    risk_score=collapse_prob,
                    current_temperature=state.temperature,
                    min_temperature=cfg.min_temperature,
                )

                # Map soft decision to legacy action format
                if decision.should_reset:
                    action = "RESET"
                elif decision.should_revert:
                    action = "REVERT"
                else:
                    action = "ACCEPT"

                new_temperature = decision.new_temperature

                intervention_metadata = decision.to_dict()
                intervention_metadata["decision_type"] = "deterministic"
                intervention_metadata["risk_score"] = collapse_prob

            if not cfg.learned_policy_shadow:
                pass  # Apply intervention
            else:
                # Shadow mode: observe but don't intervene
                action = "ACCEPT"
                new_temperature = state.temperature
                if intervention_metadata:
                    intervention_metadata["shadow_mode"] = True

    # ============================================================
    # Fallback
    # ============================================================
    else:
        pass

    return action, new_temperature, collapse_prob, intervention_metadata