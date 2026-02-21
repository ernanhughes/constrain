from ..axes.bundle import AxisBundle
from ..custom_types import Verdict
from .base import PolicyLike, get_axis, missing_to_review


class ParticipationRatioOnlyPolicy(PolicyLike):
    """
    Energy-free ablation:
      - pr <= tau_pr  => ACCEPT
      - pr >  tau_pr  => REVIEW (or REJECT if too high)
    """

    def __init__(
        self,
        *,
        tau_pr: float,
        pr_reject: float | None = None,
    ):
        self.pr_threshold = float(tau_pr)
        self.pr_reject = float(pr_reject) if pr_reject is not None else None

    @property
    def name(self) -> str:
        r = f", rej={self.pr_reject:.3f}" if self.pr_reject is not None else ""
        return f"PROnly(pr={self.pr_threshold:.3f}{r})"

    def decide(self, axes: AxisBundle, effectiveness_score: float) -> Verdict:
        pr = get_axis(axes, "participation_ratio")
        if missing_to_review(pr):
            return Verdict.REVIEW
        if self.pr_reject is not None and pr > self.pr_reject:
            return Verdict.REJECT
        return Verdict.ACCEPT if pr <= self.pr_threshold else Verdict.REVIEW

