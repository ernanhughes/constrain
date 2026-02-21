from constrain.energy.axes.bundle import AxisBundle
from constrain.energy.custom_types import Verdict

from .base import PolicyLike, get_axis, missing_to_review


class AxisFirstThenEnergyPolicy(PolicyLike):
    """
    Axis-first ablation with energy as a backstop:
      - If diagnostics indicate risk => REVIEW
      - Else fall back to energy accept/reject

    Good for: "what if we prioritize PR/sensitivity everywhere?"
    """

    def __init__(
        self,
        *,
        tau_energy: float,
        tau_pr: float,
        tau_sensitivity: float,
        tau_review: float | None = None,
    ):
        self.tau_accept = float(tau_energy)
        self.tau_review = float(tau_review) if tau_review is not None else (self.tau_accept * 1.25)
        self.pr_threshold = float(tau_pr)
        self.sensitivity_threshold = float(tau_sensitivity)

    @property
    def name(self) -> str:
        return (
            f"AxisFirst(pr={self.pr_threshold:.3f}, sens={self.sensitivity_threshold:.3f}, "
            f"tau={self.tau_accept:.3f})"
        )

    def decide(self, axes: AxisBundle, effectiveness_score: float) -> Verdict:
        pr = get_axis(axes, "participation_ratio")
        s = get_axis(axes, "sensitivity")
        e = get_axis(axes, "energy")
        if missing_to_review(pr, s, e):
            return Verdict.REVIEW

        if pr > self.pr_threshold:
            return Verdict.REVIEW
        if s > self.sensitivity_threshold:
            return Verdict.REVIEW

        if e > self.tau_review:
            return Verdict.REJECT
        return Verdict.ACCEPT if e <= self.tau_accept else Verdict.REJECT
