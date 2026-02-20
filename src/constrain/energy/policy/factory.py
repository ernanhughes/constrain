from __future__ import annotations

from typing import List, Sequence, TYPE_CHECKING

from .base import PolicyLike

if TYPE_CHECKING:
    from .energy_only import EnergyOnlyPolicy
    from .participation_ratio import ParticipationRatioOnlyPolicy
    from .sensitivity import SensitivityOnlyPolicy
    from .axes import AxisOnlyPolicy
    from .axes_energy import AxisFirstThenEnergyPolicy
    from .monotone import MonotoneAdaptivePolicy


def build_policy(
    name: str,
    *,
    tau_energy: float,
    tau_pr: float,
    tau_sensitivity: float,
    gap_width: float = 0.0,
) -> PolicyLike:
    key = name.strip().lower()

    if key in ("adaptive", "monotone_adaptive"):
        return MonotoneAdaptivePolicy(
            tau_energy=tau_energy,
            tau_pr=tau_pr,
            tau_sensitivity=tau_sensitivity,
            gap_width=gap_width,
        )

    if key in ("energy", "energy_only"):
        return EnergyOnlyPolicy(tau_energy=tau_energy)

    if key in ("pr", "pr_only", "participation_ratio"):
        return ParticipationRatioOnlyPolicy(tau_pr=tau_pr)

    if key in ("sens", "sensitivity_only"):
        return SensitivityOnlyPolicy(tau_sensitivity=tau_sensitivity)

    if key in ("axis", "axis_only"):
        return AxisOnlyPolicy(tau_pr=tau_pr, tau_sensitivity=tau_sensitivity)

    if key in ("axis_first", "axisfirst"):
        return AxisFirstThenEnergyPolicy(
            tau_energy=tau_energy,
            tau_pr=tau_pr,
            tau_sensitivity=tau_sensitivity,
        )

    raise ValueError(f"Unknown policy name: {name}")


def build_policies(
    names: Sequence[str],
    *,
    tau_energy: float,
    tau_pr: float,
    tau_sensitivity: float,
    gap_width: float = 0.0,
) -> List[PolicyLike]:
    return [
        build_policy(
            n,
            tau_energy=tau_energy,
            tau_pr=tau_pr,
            tau_sensitivity=tau_sensitivity,
            gap_width=gap_width,
        )
        for n in names
    ]
