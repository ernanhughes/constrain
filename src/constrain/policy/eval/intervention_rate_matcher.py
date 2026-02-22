from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class InterventionRateMatcher:
    """Deterministic random intervention schedule that matches a reference rate."""

    intervention_rate: float

    @classmethod
    def from_reference_trials(cls, reference_trials: List[Dict[str, Any]]) -> "InterventionRateMatcher":
        total_turns = sum(int(t["failure_turn"]) for t in reference_trials)
        total_interventions = sum(int(t.get("interventions", 0)) for t in reference_trials)
        rate = (total_interventions / total_turns) if total_turns > 0 else 0.0
        return cls(intervention_rate=float(rate))

    def should_intervene(self, *, turn: int, seed: int) -> bool:
        rng = np.random.default_rng(seed + (turn * 7919))
        return bool(rng.random() < self.intervention_rate)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump({"intervention_rate": self.intervention_rate}, f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "InterventionRateMatcher":
        path = Path(path)
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(intervention_rate=float(data["intervention_rate"]))
