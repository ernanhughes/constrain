from dataclasses import dataclass
from typing import List


@dataclass
class ControlExperimentConfig:
    num_problems: int = 200
    seeds: List[int] = (42, 43, 44, 45, 46)
    max_iterations: int = 4
    policies: List[str] = ("baseline", "energy", "learned")
    shadow_learned: bool = True
