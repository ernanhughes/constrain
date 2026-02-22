# constrain/policy/threshold_result.py

from dataclasses import dataclass
from constrain.policy.custom_types import Thresholds


@dataclass
class ThresholdResult:
    thresholds: Thresholds
    sample_count: int