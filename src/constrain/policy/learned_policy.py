# constrain/policy/learned_policy.py
from __future__ import annotations

import joblib
import numpy as np
from typing import Optional, Dict, Tuple

class LearnedPolicy:
    def __init__(
        self,
        base_path: str,
        weights: Optional[Dict[str, float]] = None,
        bias: float = 0.0,
    ):
        """
        Load 3-head policy models.
        
        Args:
            base_path: Path without suffix (e.g., "models/policy_head")
            weights: Dict with keys "collapse", "utility", "delta"
            bias: Decision threshold on value score
        """
        # Remove .joblib suffix if present
        base_path = base_path.replace(".joblib", "")

        self.collapse_model = joblib.load(f"{base_path}_collapse.joblib")
        self.utility_model = joblib.load(f"{base_path}_utility.joblib")
        self.delta_model = joblib.load(f"{base_path}_delta.joblib")

        # Get feature names from first model
        self.feature_names = self.collapse_model.feature_names_in_.tolist()

        # Default weights (tune via sweep)
        self.weights = weights or {
            "collapse": 1.0,
            "utility": 1.0,
            "delta": 1.0,
        }
        self.bias = bias

    def predict(self, feature_dict: Dict[str, float]) -> Dict[str, float]:
        """Get predictions from all 3 heads."""
        X = np.array([[feature_dict.get(f, 0.0) for f in self.feature_names]])

        collapse_prob = float(self.collapse_model.predict_proba(X)[0, 1])
        utility_prob = float(self.utility_model.predict_proba(X)[0, 1])
        delta_value = float(self.delta_model.predict(X)[0])

        return {
            "collapse_prob": collapse_prob,
            "utility_prob": utility_prob,
            "delta_value": delta_value,
        }

    def decide(
        self,
        feature_dict: Dict[str, float],
    ) -> Tuple[str, Dict[str, float], float]:
        """
        Make intervention decision using value function.
        
        Value = w_collapse * collapse_prob + w_utility * utility_prob - w_delta * delta_value
        
        Returns:
            action: "REVERT" or "ACCEPT"
            preds: Dict with all 3 head outputs
            value: Computed value score
        """
        preds = self.predict(feature_dict)

        # Value function for INTERVENTION
        # High collapse → intervene
        # High utility → intervene
        # High delta (natural improvement) → don't intervene
        value = (
            self.weights["collapse"] * preds["collapse_prob"]
            + self.weights["utility"] * preds["utility_prob"]
            - self.weights["delta"] * preds["delta_value"]
        )

        action = "REVERT" if value > self.bias else "ACCEPT"

        return action, preds, value