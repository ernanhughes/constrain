# constrain/policy/policy_config.py

from dataclasses import dataclass


@dataclass
class PolicyConfig:
    min_temperature: float
    revert_cooldown_factor: float
    aggressive_cooldown_factor: float
    reset_cooldown_factor: float

    random_revert_probability: float

    geometry_pr_threshold: float
    geometry_sensitivity_threshold: float
    geometry_band_width_factor: float