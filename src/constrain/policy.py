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


def apply_policy(policy_id, energy, reasoning, last_stable, prompt, temperature, memory: Memory, run_id: str):

    cfg = get_config()

    tau_soft, tau_medium, tau_hard = _get_thresholds(cfg, memory, run_id)

    # -------------------------------------------------
    # Accept-all baseline
    # -------------------------------------------------

    if policy_id == 0:
        return reasoning, temperature, "ACCEPT"

    # -------------------------------------------------
    # Energy below soft threshold = stable
    # -------------------------------------------------

    if energy <= tau_soft:
        return reasoning, temperature, "ACCEPT"

    # -------------------------------------------------
    # Policy variants
    # -------------------------------------------------

    if policy_id == 1:
        return last_stable, temperature, "REVERT"

    if policy_id == 2:
        return last_stable, temperature * 0.9, "REVERT_COOL"

    if policy_id == 3:
        if energy > tau_medium:
            return last_stable, temperature * 0.75, "REVERT_AGGRESSIVE"
        return last_stable, temperature, "REVERT"

    if policy_id == 4:
        if energy > tau_hard:
            return prompt, temperature * 0.7, "RESET_PROMPT"
        return last_stable, temperature, "REVERT"

    if policy_id == 5:
        if energy > tau_medium:
            return prompt, temperature * 0.85, "RESET"
        return last_stable, temperature * 0.85, "REVERT_STABILIZE"

    if policy_id == 6:
        import random

        # Only consider intervention if above soft threshold
        if energy <= tau_soft:
            return reasoning, temperature, "ACCEPT"

        # Randomly select from reasonable actions
        actions = [
            ("REVERT", 0.4),
            ("REVERT_COOL", 0.3),
            ("RESET", 0.2),
            ("RESET_PROMPT", 0.1),
        ]

        r = random.random()
        cumulative = 0
        for action, weight in actions:
            cumulative += weight
            if r <= cumulative:
                chosen = action
                break


        if chosen == "REVERT":
            return last_stable, temperature, "REVERT_RANDOM"

        if chosen == "REVERT_COOL":
            return last_stable, max(0.1, temperature * 0.85), "REVERT_COOL_RANDOM"

        if chosen == "RESET":
            return prompt, max(0.1, temperature * 0.85), "RESET_RANDOM"

        if chosen == "RESET_PROMPT":
            return prompt, max(0.1, temperature * 0.7), "RESET_PROMPT_RANDOM"


    return reasoning, temperature, "ACCEPT"
