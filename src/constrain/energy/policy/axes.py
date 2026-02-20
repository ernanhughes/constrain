from constrain.energy.axes.bundle import AxisBundle
from constrain.energy.custom_types import Verdict
from .base import PolicyLike

class AxisOnlyPolicy(PolicyLike):
    """
    Energy-free composite ablation:
      - if pr high or sens high => REVIEW (or REJECT if extreme)
      - else                   => ACCEPT

    This is the cleanest way to see whether PR/sensitivity carry signal at all,
    without energy soaking the whole experiment.
    """

    def __init__(
        self,
        *,
        tau_pr: float,
        tau_sensitivity: float,
        pr_reject: float | None = None,
        sens_reject: float | None = None,
    ):
        self.pr_threshold = float(tau_pr)
        self.sensitivity_threshold = float(tau_sensitivity)
        self.pr_reject = float(pr_reject) if pr_reject is not None else None
        self.sens_reject = float(sens_reject) if sens_reject is not None else None

    @property
    def name(self) -> str:
        return (
            f"AxisOnly(pr={self.pr_threshold:.3f}, sens={self.sensitivity_threshold:.3f})"
        )

    def decide(self, axes: AxisBundle, effectiveness_score: float) -> Verdict:
        pr = _get_axis(axes, "participation_ratio")
        s = _get_axis(axes, "sensitivity")
        if _missing_to_review(pr, s):
            return Verdict.REVIEW

        if self.pr_reject is not None and pr > self.pr_reject:
            return Verdict.REJECT
        if self.sens_reject is not None and s > self.sens_reject:
            return Verdict.REJECT

        if pr > self.pr_threshold:
            return Verdict.REVIEW
        if s > self.sensitivity_threshold:
            return Verdict.REVIEW
        return Verdict.ACCEPT

