# constrain/policy/learned_policy.py
from __future__ import annotations

from typing import Dict, Tuple

import joblib
import numpy as np


class LearnedPolicy:
    """
    Causal Learned Policy using S-Learner outcome model.

    Loads:
        *_outcome.joblib
        *_features.joblib

    Decision rule:
        τ(X) = P(collapse | accept) - P(collapse | intervene)

        If τ(X) > bias → INTERVENE
        Else → ACCEPT
    """

    def __init__(
        self,
        model_path: str,
        bias: float = 0.0,
    ):
        model_path = model_path.replace(".joblib", "")

        self.outcome_model = joblib.load(f"{model_path}_outcome.joblib")
        self.feature_names = joblib.load(f"{model_path}_features.joblib")

        self.bias = bias

        if "is_intervention" not in self.feature_names:
            raise ValueError(
                "Model missing 'is_intervention' feature — incompatible model."
            )


    # ------------------------------------------------------------
    # Internal feature builder
    # ------------------------------------------------------------
    def _build_feature_vector(
        self,
        feature_dict: Dict[str, float],
        is_intervention: int,
    ):
        row = []
        for f in self.feature_names:
            if f == "is_intervention":
                row.append(is_intervention)
            else:
                row.append(feature_dict.get(f, 0.0))
        return np.array([row])

    # ------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------
    def predict(
        self,
        feature_dict: Dict[str, float],
    ) -> Dict[str, float]:
        """
        Compute counterfactual predictions.
        """

        # ACCEPT scenario (A=0)
        X_accept = self._build_feature_vector(feature_dict, 0)
        p_accept = float(self.outcome_model.predict_proba(X_accept)[0, 1])

        # INTERVENE scenario (A=1)
        X_intervene = self._build_feature_vector(feature_dict, 1)
        p_intervene = float(self.outcome_model.predict_proba(X_intervene)[0, 1])

        treatment_effect = p_accept - p_intervene

        return {
            "p_accept": p_accept,
            "p_intervene": p_intervene,
            "treatment_effect": treatment_effect,
        }

    # ------------------------------------------------------------
    # Decision
    # ------------------------------------------------------------
    def decide(
        self,
        feature_dict: Dict[str, float],
    ) -> Tuple[str, Dict[str, float], float]:
        """
        Decide whether to intervene.

        Intervene if:
            treatment_effect > bias
        """

        preds = self.predict(feature_dict)

        value = preds["treatment_effect"]

        action = "REVERT" if value > self.bias else "ACCEPT"

        return action, preds, value