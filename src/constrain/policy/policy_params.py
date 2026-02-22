from dataclasses import dataclass

@dataclass
class PolicyParams:
    min_temperature: float
    revert_cooldown_factor: float
    aggressive_cooldown_factor: float
    reset_cooldown_factor: float