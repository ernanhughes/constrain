# constrain/policy/learned_policy.py

from __future__ import annotations

from typing import Dict, Tuple

import joblib
import numpy as np


class LearnedPolicy:
    """
    XGBoost-based collapse prediction policy.

    This class:
    - Loads trained model
    - Aligns features safely
    - Handles missing features
    - Returns (action, probability)
    """

    def __init__(self, model_path: str, threshold: float = 0.5):
        self.model = joblib.load(model_path)
        self.threshold = threshold

        # Capture feature names from trained model
        if hasattr(self.model, "feature_names_in_"):
            self.feature_names = list(self.model.feature_names_in_)
        else:
            # Fallback if not available
            raise ValueError(
                "Trained model does not expose feature_names_in_. "
                "Ensure model was trained with pandas DataFrame input."
            )

    # ---------------------------------------------------------
    # Core Decision Logic
    # ---------------------------------------------------------

    def decide(self, feature_dict: Dict[str, float]) -> Tuple[str, float]:
        """
        Given feature dict, return (action, probability).
        """

        # Build ordered feature vector matching training schema
        row = []

        for f in self.feature_names:
            val = feature_dict.get(f, 0.0)

            try:
                val = float(val)
            except Exception:
                val = 0.0

            row.append(val)

        X = np.array([row], dtype=np.float32)

        prob = float(self.model.predict_proba(X)[0, 1])

        if prob > self.threshold:
            return "REVERT", prob

        return "ACCEPT", prob

    # ---------------------------------------------------------
    # Optional: threshold tuning hook
    # ---------------------------------------------------------

    def set_threshold(self, new_threshold: float):
        self.threshold = float(new_threshold)
