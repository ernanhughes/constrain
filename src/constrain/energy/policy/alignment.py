
from constrain.energy.axes.bundle import AxisBundle
from constrain.energy.custom_types import Verdict

from .base import PolicyLike, get_axis, missing_to_review


class AlignmentOnlyPolicy(PolicyLike):
    """
    Energy-free ablation using 'explained' (alignment) if present:
      - explained >= tau_align => ACCEPT
      - else                  => REVIEW/REJECT
    Notes:
      - If you don't currently store 'explained', wire it as an axis (0..1).
    """

    def __init__(self, *, tau_alignment: float, align_reject: float | None = None):
        self.align_threshold = float(tau_alignment)
        self.align_reject = float(align_reject) if align_reject is not None else None

    @property
    def name(self) -> str:
        r = f", rej={self.align_reject:.3f}" if self.align_reject is not None else ""
        return f"AlignOnly(align={self.align_threshold:.3f}{r})"

    def decide(self, axes: AxisBundle, effectiveness_score: float) -> Verdict:
        a = get_axis(axes, "explained")
        if missing_to_review(a):
            return Verdict.REVIEW
        if self.align_reject is not None and a < self.align_reject:
            return Verdict.REJECT
        return Verdict.ACCEPT if a >= self.align_threshold else Verdict.REVIEW

