# policy.py

from constrain.config import get_config
from constrain.data.memory import Memory
from constrain.calibration.recursive import RecursiveCalibrator

def _get_thresholds(cfg, memory: Memory, run_id: str):
    """
    Select thresholds based on policy mode.
    """

    if cfg.policy_mode == "static":
        return cfg.tau_soft, cfg.tau_medium, cfg.tau_hard

    if cfg.policy_mode == "recursive":
        return RecursiveCalibrator.get_thresholds(memory, run_id)
    
    if cfg.policy_mode == "adaptive":
        raise NotImplementedError("Adaptive policy mode not implemented yet.")

    if cfg.policy_mode == "dynamic":
        raise NotImplementedError("Dynamic policy mode not implemented yet.")

    else:
        raise ValueError(f"Unknown policy_mode: {cfg.policy_mode}")




def apply_policy(
    policy_id: int,
    axes: dict,
    reasoning: str,
    state,
    memory: Memory,
    run_id: str,
):
    """
    New policy interface.

    Returns:
        (action: str, new_temperature: float)
    """

    cfg = get_config()

    energy = axes.get("energy")
    pr = axes.get("participation_ratio")
    sensitivity = axes.get("sensitivity")

    temperature = state.temperature

    tau_soft = cfg.tau_soft
    tau_medium = cfg.tau_medium
    tau_hard = cfg.tau_hard

    # -------------------------------------------------
    # Policy 0 — Accept All (Baseline)
    # -------------------------------------------------

    if policy_id == 0:
        return "ACCEPT", temperature

    # -------------------------------------------------
    # If stable region → ACCEPT
    # -------------------------------------------------

    if energy <= tau_soft:
        return "ACCEPT", temperature

    # -------------------------------------------------
    # Policy 1 — Simple Revert
    # -------------------------------------------------

    if policy_id == 1:
        return "REVERT", temperature

    # -------------------------------------------------
    # Policy 2 — Revert + Cool
    # -------------------------------------------------

    if policy_id == 2:
        return "REVERT", max(0.1, temperature * 0.9)

    # -------------------------------------------------
    # Policy 3 — Aggressive Revert
    # -------------------------------------------------

    if policy_id == 3:
        if energy > tau_medium:
            return "REVERT", max(0.1, temperature * 0.75)
        return "REVERT", temperature

    # -------------------------------------------------
    # Policy 4 — Hard Reset on Extreme Energy
    # -------------------------------------------------

    if policy_id == 4:
        if energy > tau_hard:
            return "RESET", max(0.1, temperature * 0.7)
        return "REVERT", temperature

    # -------------------------------------------------
    # Policy 5 — Medium Reset Strategy
    # -------------------------------------------------

    if policy_id == 5:
        if energy > tau_medium:
            return "RESET", max(0.1, temperature * 0.85)
        return "REVERT", max(0.1, temperature * 0.85)

    # -------------------------------------------------
    # Policy 6 — Simple Random
    # -------------------------------------------------

    if policy_id == 6:
        import random
        if random.random() < 0.3:
            return "REVERT", temperature
        return "ACCEPT", temperature

    # -------------------------------------------------
    # Policy 7 — Weighted Random
    # -------------------------------------------------

    if policy_id == 7:
        import random

        if energy <= tau_soft:
            return "ACCEPT", temperature

        r = random.random()

        if r < 0.4:
            return "REVERT", temperature
        elif r < 0.7:
            return "REVERT", max(0.1, temperature * 0.85)
        elif r < 0.9:
            return "RESET", max(0.1, temperature * 0.85)
        else:
            return "RESET", max(0.1, temperature * 0.7)

    # -------------------------------------------------
    # Policy 8 — Geometry-Aware Ambiguity Band
    # -------------------------------------------------

    if policy_id == 8:
        gap_width = 0.15 * tau_soft
        low = tau_soft - gap_width
        high = tau_soft + gap_width

        # Clearly good
        if energy <= low:
            return "ACCEPT", temperature

        # Clearly bad
        if energy >= high:
            return "REJECT", temperature

        # Ambiguity band → geometry activation
        if pr > 0.6:
            return "REVERT", temperature

        if sensitivity > 0.5:
            return "REVERT", temperature

        if energy <= tau_soft:
            return "ACCEPT", temperature

        return "REVERT", temperature

    # -------------------------------------------------
    # Default fallback
    # -------------------------------------------------

    return "ACCEPT", temperature
