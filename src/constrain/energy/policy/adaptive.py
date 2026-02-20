
from constrain.energy.axes.bundle import AxisBundle
from constrain.energy.custom_types import Verdict


class AdaptivePolicy:

    def __init__(
        self,
        *,
        tau_energy: float,
        tau_pr: float,
        tau_sensitivity: float,
        tau_review: float | None = None,
        hard_negative_gap: float = 0.0,
        gap_width: float = 0.1,
    ):
        self.tau_accept = tau_energy
        self.tau_review = tau_review or (tau_energy * 1.25)
        self.hard_negative_gap = hard_negative_gap
        self.gap_width = gap_width
        self.pr_threshold = tau_pr
        self.sensitivity_threshold = tau_sensitivity

        self.thresholds = {
            "participation_ratio": tau_pr,
            "sensitivity": tau_sensitivity,
        }

    @property
    def name(self) -> str:
        return f"AdaptivePolicy(tau_energy={self.tau_accept:.2f}, tau_pr={self.pr_threshold:.2f}, tau_sensitivity={self.sensitivity_threshold:.2f})"

    def decide(
        self,
        axes: AxisBundle,
        effectiveness_score: float
    ) -> Verdict:

        energy = axes.get("energy")

        # Hard reject zone (always)
        if energy > self.tau_review:
            return Verdict.REJECT

        # Define ambiguity band width (energy-relative)
        gap_width = self.gap_width * self.tau_accept
        low = self.tau_accept - gap_width
        high = self.tau_accept + gap_width

        # -------------------------------------------------
        # Region 1 — Clearly Good (Energy Dominates)
        # -------------------------------------------------
        if energy <= low:
            return Verdict.ACCEPT

        # -------------------------------------------------
        # Region 2 — Clearly Bad (Energy Dominates)
        # -------------------------------------------------
        if energy >= high:
            return Verdict.REJECT

        # -------------------------------------------------
        # Region 3 — Ambiguity Band (Geometry Activated)
        # -------------------------------------------------

        # Participation constraint
        if axes.get("participation_ratio") > self.pr_threshold:
            return Verdict.REVIEW

        # Sensitivity constraint
        if axes.get("sensitivity") > self.sensitivity_threshold:
            return Verdict.REVIEW

        # Monotone: never upgrade above tau_accept
        if energy <= self.tau_accept:
            return Verdict.ACCEPT
        return Verdict.REVIEW
