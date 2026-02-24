from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
except Exception:  # pragma: no cover
    torch = None
    nn = None


@dataclass
class ActionBounds:
    tau_min: float = 0.0
    tau_max: float = 10.0
    temp_min: float = 0.0
    temp_max: float = 2.0
    recursion_min: int = 1
    recursion_max: int = 8


@dataclass
class AdapterObservation:
    total_energy: float
    delta_energy: float
    grounding_energy: float = 0.0
    stability_energy: float = 0.0
    entropy: float = 0.0
    iteration_progress: float = 0.0
    phase_value: float = 0.0
    recent_intervention_rate: float = 0.0

    def to_tensor(self):
        if torch is None:
            raise RuntimeError("torch not available")
        return torch.tensor([
            self.total_energy, self.delta_energy,
            self.grounding_energy, self.stability_energy,
            self.entropy, self.iteration_progress,
            self.phase_value, self.recent_intervention_rate
        ], dtype=torch.float32)


@dataclass
class AdapterAction:
    tau_hard: float
    temperature: float
    recursion_limit: int

    def clamp(self, b: ActionBounds) -> "AdapterAction":
        return AdapterAction(
            tau_hard=float(np.clip(self.tau_hard, b.tau_min, b.tau_max)),
            temperature=float(np.clip(self.temperature, b.temp_min, b.temp_max)),
            recursion_limit=int(np.clip(self.recursion_limit, b.recursion_min, b.recursion_max)),
        )

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


@dataclass
class SafetyGateResult:
    action: AdapterAction
    fallback_triggered: bool
    reason: str
    lyapunov_ok: bool
    smoothed: bool


class RLParameterAdapter:
    """
    Minimal adapter wrapper.

    This is intentionally light so it can be swapped with your full implementation.
    """
    def __init__(self, bounds: Optional[ActionBounds] = None, checkpoint_path: Optional[str] = None):
        self.bounds = bounds or ActionBounds()
        self.last_action: Optional[AdapterAction] = None
        self.checkpoint_path = checkpoint_path

    def propose_action_ungated(self, obs: AdapterObservation) -> AdapterAction:
        # Placeholder heuristic: tighten when energy rising
        tau = 7.0 if obs.delta_energy > 0 else 8.5
        temp = 0.5 if obs.delta_energy > 0 else 1.0
        rec = 2 if obs.delta_energy > 0 else 4
        return AdapterAction(tau_hard=tau, temperature=temp, recursion_limit=rec).clamp(self.bounds)

    def get_safe_action(self, obs: AdapterObservation, energy_history: List[float]) -> SafetyGateResult:
        proposed = self.propose_action_ungated(obs)
        # Simple Lyapunov-ish: if recent mean exploding, fallback.
        ly_ok = True
        reason = "ok"
        if len(energy_history) >= 20:
            old = float(np.mean(energy_history[-20:-10]))
            new = float(np.mean(energy_history[-10:]))
            if old > 0 and new > old * 2.0:
                ly_ok = False
        if not ly_ok:
            proposed = AdapterAction(tau_hard=9.0, temperature=0.2, recursion_limit=2).clamp(self.bounds)
            return SafetyGateResult(proposed, True, "lyapunov_violation", False, False)
        return SafetyGateResult(proposed, False, reason, True, False)
