from constrain.energy.axes.bundle import AxisBundle
from constrain.energy.custom_types import Verdict
from .base import PolicyLike, get_axis, missing_to_review

class MonotoneAdaptivePolicy(PolicyLike):
    """
    Energy-first, but monotone:
      - energy decides ACCEPT/REJECT
      - diagnostics can only downgrade ACCEPT -> REVIEW (or REJECT if extreme)
      - diagnostics never upgrade REJECT -> ACCEPT

    This prevents the "adaptive policy FAR explosion" you saw on hard-mined.
    """

    def __init__(
        self,
        *,
        tau_energy: float,
        tau_pr: float,
        tau_sensitivity: float,
        tau_review: float | None = None,
        pr_reject: float | None = None,
        sens_reject: float | None = None,
        gap_width: float = 0.0,  # 0 => always consider diagnostics
    ):
        self.tau_accept = float(tau_energy)
        self.tau_review = float(tau_review) if tau_review is not None else (self.tau_accept * 1.25)
        self.pr_threshold = float(tau_pr)
        self.sensitivity_threshold = float(tau_sensitivity)
        self.pr_reject = float(pr_reject) if pr_reject is not None else None
        self.sens_reject = float(sens_reject) if sens_reject is not None else None
        self.gap_width = float(gap_width)

    @property
    def name(self) -> str:
        return (
            f"MonotoneAdaptive(tau={self.tau_accept:.3f}, "
            f"gap={self.gap_width:.3f}, pr={self.pr_threshold:.3f}, "
            f"sens={self.sensitivity_threshold:.3f})"
        )

    def decide(self, axes: AxisBundle, effectiveness_score: float) -> Verdict:
        e = get_axis(axes, "energy")
        if missing_to_review(e):
            return Verdict.REVIEW

        # Hard reject zone (energy dominates)
        if e > self.tau_review:
            return Verdict.REJECT

        # Base energy decision
        base = Verdict.ACCEPT if e <= self.tau_accept else Verdict.REJECT
        if base != Verdict.ACCEPT:
            return base  # monotone: do not upgrade

        # Diagnostics are only consulted globally, or inside a band (if gap_width > 0)
        if self.gap_width > 0.0 and abs(e - self.tau_accept) > self.gap_width:
            return Verdict.ACCEPT

        pr = get_axis(axes, "participation_ratio")
        s = get_axis(axes, "sensitivity")
        if missing_to_review(pr, s):
            return Verdict.REVIEW

        # Extreme reject (optional)
        if self.pr_reject is not None and pr > self.pr_reject:
            return Verdict.REJECT
        if self.sens_reject is not None and s > self.sens_reject:
            return Verdict.REJECT

        # Soft downgrade to REVIEW
        if pr > self.pr_threshold:
            return Verdict.REVIEW
        if s > self.sensitivity_threshold:
            return Verdict.REVIEW

        return Verdict.ACCEPT
