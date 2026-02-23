# constrain/policy/randomized_exploration_policy.py
"""
RANDOMIZED EXPLORATION FOR CAUSAL DATA COLLECTION

When risk > threshold:
  50% intervene
  50% accept

This creates clean causal data for future model training.
"""

import numpy as np
import joblib
from constrain.config import get_config


class RandomizedExplorationPolicy:
    def __init__(
        self,
        model_path: str = None,
        risk_threshold: float = 0.7,
        exploration_rate: float = 0.5,
    ):
        cfg = get_config()
        
        if model_path is None:
            model_path = cfg.learned_model_path.replace(".joblib", "_outcome.joblib")
        
        self.model = joblib.load(model_path)
        self.feature_cols = joblib.load(
            model_path.replace("_outcome.joblib", "_features.joblib")
        )
        self.risk_threshold = risk_threshold
        self.exploration_rate = exploration_rate

    def decide(self, step_features: dict) -> tuple[str, dict]:
        """
        Returns:
            action: "ACCEPT" or "INTERVENE"
            metadata: decision info for logging
        """
        # Prepare features
        X = np.array([[step_features.get(k, 0.0) for k in self.feature_cols]])

        # Predict collapse risk (with A=0 baseline)
        X_baseline = X.copy()
        X_baseline[0, self.feature_cols.index("is_intervention")] = 0
        risk = self.model.predict_proba(X_baseline)[0][1]

        # Decision
        if risk > self.risk_threshold:
            # High risk — randomize for causal data
            if np.random.random() < self.exploration_rate:
                action = "INTERVENE"
                decision_type = "exploration"
            else:
                action = "ACCEPT"
                decision_type = "exploration"
        else:
            # Low risk — follow model recommendation
            X_intervene = X.copy()
            X_intervene[0, self.feature_cols.index("is_intervention")] = 1
            p_accept = self.model.predict_proba(X_baseline)[0][1]
            p_intervene = self.model.predict_proba(X_intervene)[0][1]

            if p_accept > p_intervene:
                action = "INTERVENE"
            else:
                action = "ACCEPT"
            decision_type = "exploitation"

        metadata = {
            "risk": float(risk),
            "decision_type": decision_type,
            "action": action,
        }

        return action, metadata