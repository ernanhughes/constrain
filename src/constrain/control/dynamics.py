"""
Energy dynamics computations - pure, global, policy-independent.

All functions are stateless and deterministic.
"""
from typing import Tuple, Optional
from .types import ViolationLevel


def classify_violation(
    energy: float,
    tau_soft: float,
    tau_medium: float,
    tau_hard: float,
) -> ViolationLevel:
    """
    Classify violation level from energy value.
    
    Global definition - same for all policies.
    """
    if energy > tau_hard:
        return ViolationLevel.HARD
    elif energy > tau_medium:
        return ViolationLevel.MEDIUM
    elif energy > tau_soft:
        return ViolationLevel.SOFT
    return ViolationLevel.NONE


def compute_slope(energy: float, prev_energy: Optional[float]) -> float:
    """
    Compute energy slope (delta from previous step).
    
    Returns 0.0 if no previous energy (first step).
    """
    if prev_energy is None:
        return 0.0
    return energy - prev_energy


def update_counters(
    violation: ViolationLevel,
    slope: float,
    prev_hard: int,
    prev_rising: int,
    prev_energy: Optional[float],
) -> Tuple[int, int]:
    """
    Update consecutive violation counters.
    
    Returns: (consecutive_hard, consecutive_rising)
    
    Rules:
    - consecutive_hard: increments only if violation==HARD, else resets to 0
    - consecutive_rising: increments only if slope > 0 AND prev_energy is not None, else resets to 0
    """
    # Update hard counter
    if violation == ViolationLevel.HARD:
        consecutive_hard = prev_hard + 1
    else:
        consecutive_hard = 0
    
    # Update rising counter (Clarification #3: first step cannot be "rising")
    if prev_energy is not None and slope > 0:
        consecutive_rising = prev_rising + 1
    else:
        consecutive_rising = 0
    
    return consecutive_hard, consecutive_rising


def is_collapse(
    violation: ViolationLevel,
    consecutive_hard: int,
    consecutive_rising: int,
    collapse_hard_n: int = 3,
    collapse_rising_n: int = 2,
) -> bool:
    """
    Determine if collapse has occurred.
    
    Collapse = persistent high AND rising energy (containment failure).
    
    Global definition - same for all policies.
    """
    if violation != ViolationLevel.HARD:
        return False
    
    if consecutive_hard < collapse_hard_n:
        return False
    
    if consecutive_rising < collapse_rising_n:
        return False
    
    return True


def reset_dynamics() -> Tuple[Optional[float], int, int]:
    """
    Reset all dynamics tracking (called on RESET action).
    
    Returns: (prev_energy, consecutive_hard, consecutive_rising)
    """
    return None, 0, 0