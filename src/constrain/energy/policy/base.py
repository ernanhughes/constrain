from ..axes.bundle import AxisBundle
from ..custom_types import Verdict

from typing import Optional

class PolicyLike:
    tau_accept: float | None = None
    tau_review: float | None = None
    pr_threshold: float | None = None
    sensitivity_threshold: float | None
    hard_negative_gap: float = 0.0

    @property
    def name(self) -> str:  # pragma: no cover
        raise NotImplementedError

    def decide(self, axes: AxisBundle, effectiveness_score: float) -> Verdict:  # pragma: no cover
        raise NotImplementedError


def get_axis(axes: AxisBundle, key: str) -> Optional[float]:
    try:
        v = axes.get(key)
    except Exception:
        return None
    if v is None:
        return None
    try:
        return float(v)
    except Exception:
        return None


def missing_to_review(*vals: Optional[float]) -> bool:
    return any(v is None for v in vals)
