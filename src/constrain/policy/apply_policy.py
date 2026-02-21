# constrain/policy/apply_policy.py

from __future__ import annotations

import os
import random
from typing import Dict, Optional, Tuple

from constrain.calibration.recursive import RecursiveCalibrator
from constrain.config import get_config
from constrain.data.memory import Memory
from constrain.policy.learned_policy import LearnedPolicy
from constrain.reasoning_state import ReasoningState

_learned_policy_instance = None


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
        # 🔥 critical: update threshold dynamically
        if threshold is not None:
            _learned_policy_instance.set_threshold(threshold)

    return _learned_policy_instance

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
    threshold: float = None
) -> Tuple[str, float, Optional[float]]:

    cfg = get_config()

    # Default outputs
    action = "ACCEPT"
    new_temperature = state.temperature
    collapse_prob: Optional[float] = None

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
    # Policy 1 — Simple Revert
    # ============================================================

    elif policy_id == 1:
        action = "REVERT"

    # ============================================================
    # Policy 2 — Revert + Cool
    # ============================================================

    elif policy_id == 2:
        action = "REVERT"
        new_temperature = max(0.1, state.temperature * 0.9)

    # ============================================================
    # Policy 3 — Aggressive Revert
    # ============================================================

    elif policy_id == 3:
        action = "REVERT"
        if energy > tau_medium:
            new_temperature = max(0.1, state.temperature * 0.75)

    # ============================================================
    # Policy 4 — Hard Reset
    # ============================================================

    elif policy_id == 4:
        if energy > tau_hard:
            action = "RESET"
            new_temperature = max(0.1, state.temperature * 0.7)
        else:
            action = "REVERT"

    # ============================================================
    # Policy 5 — Medium Reset
    # ============================================================

    elif policy_id == 5:
        if energy > tau_medium:
            action = "RESET"
        else:
            action = "REVERT"

        new_temperature = max(0.1, state.temperature * 0.85)

    # ============================================================
    # Policy 6 — Random
    # ============================================================

    elif policy_id == 6:
        if random.random() < 0.3:
            action = "REVERT"

    # ============================================================
    # Policy 7 — Weighted Random
    # ============================================================

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

    # ============================================================
    # Policy 8 — Geometry Band
    # ============================================================

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
    # Policy 99 — Learned Policy
    # ============================================================

    elif policy_id == 99:

        learned_policy = _get_learned_policy(threshold=threshold)

        if learned_policy is not None:

            action, collapse_prob = learned_policy.decide(flat_metrics)
            print("DEBUG collapse_prob:", collapse_prob)

            if not cfg.learned_policy_shadow:
                if action == "REVERT":
                    new_temperature = max(0.1, state.temperature * 0.9)
            else:
                # Shadow mode: do not intervene
                action = "ACCEPT"

    # ============================================================
    # Fallback
    # ============================================================

    else:
        pass

    return action, new_temperature, collapse_prob
