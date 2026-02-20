from ..axes.bundle import AxisBundle
from ..custom_types import Verdict
from .base import PolicyLike, get_axis, missing_to_review


class SensitivityOnlyPolicy(PolicyLike):
    """
    Energy-free ablation:
      - sens <= tau_sens => ACCEPT
      - sens >  tau_sens => REVIEW (or REJECT if too high)
    """

    def __init__(
        self,
        *,
        tau_sensitivity: float,
        sens_reject: float | None = None,
    ):
        self.sensitivity_threshold = float(tau_sensitivity)
        self.sens_reject = float(sens_reject) if sens_reject is not None else None

    @property
    def name(self) -> str:
        r = f", rej={self.sens_reject:.3f}" if self.sens_reject is not None else ""
        return f"SensOnly(sens={self.sensitivity_threshold:.3f}{r})"

    def decide(self, axes: AxisBundle, effectiveness_score: float) -> Verdict:
        s = get_axis(axes, "sensitivity")
        if missing_to_review(s):
            return Verdict.REVIEW
        if self.sens_reject is not None and s > self.sens_reject:
            return Verdict.REJECT
        return Verdict.ACCEPT if s <= self.sensitivity_threshold else Verdict.REVIEW
