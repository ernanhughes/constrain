# constrain/policy/randomized_causal_policy.py
"""
Randomized Causal Policy

When risk is high, randomize action to create:
    A ⟂ potential_outcome | state

This enables clean causal effect estimation.
"""

from typing import Dict, Optional, Tuple

import joblib
import numpy as np

from constrain.config import get_config


class RandomizedCausalPolicy:
    """
    Policy that randomizes intervention decisions in high-risk states.

    Creates locally randomized data for causal inference.
    """

    def __init__(
        self,
        risk_model_path: Optional[str] = None,
        risk_threshold: float = 0.7,
        randomization_rate: float = 0.5,
        seed: Optional[int] = None,
    ):
        cfg = get_config()

        if risk_model_path is None:
            risk_model_path = cfg.learned_model_path.replace(".joblib", "_outcome.joblib")

        self.risk_model = joblib.load(risk_model_path)
        self.feature_cols = joblib.load(
            risk_model_path.replace("_outcome.joblib", "_features.joblib")
        )

        self.risk_threshold = risk_threshold
        self.randomization_rate = randomization_rate
        self.rng = np.random.RandomState(seed)

    def decide(self, feature_dict: Dict[str, float], problem_seed: int = 0) -> Tuple[str, Dict]:
        """
        Make intervention decision with optional randomization.

        Args:
            feature_dict: State features
            problem_seed: Seed for reproducibility per problem

        Returns:
            action: "ACCEPT" or "INTERVENE"
            metadata: Decision info for logging
        """
        # Prepare feature vector
        X = np.array([[feature_dict.get(f, 0.0) for f in self.feature_cols]])

        # Ensure is_intervention is 0 for risk prediction (counterfactual)
        if "is_intervention" in self.feature_cols:
            idx = self.feature_cols.index("is_intervention")
            X[0, idx] = 0

        # Predict collapse risk
        risk = float(self.risk_model.predict_proba(X)[0, 1])

        # Decision logic
        if risk > self.risk_threshold:
            # HIGH RISK — RANDOMIZE
            problem_rng = np.random.RandomState(problem_seed)
            if problem_rng.random() < self.randomization_rate:
                action = "INTERVENE"
            else:
                action = "ACCEPT"
            decision_type = "randomized"
        else:
            # LOW RISK — ACCEPT
            action = "ACCEPT"
            decision_type = "low_risk"

        metadata = {
            "risk_score": risk,
            "decision_type": decision_type,
            "risk_threshold": self.risk_threshold,
            "randomization_rate": self.randomization_rate,
            "action": action,
        }

        return action, metadata

    def is_randomized_decision(self, metadata: Dict) -> bool:
        """Check if this decision was randomized (for filtering later)."""
        return metadata.get("decision_type") == "randomized"


def main():
    """Test the policy with sample features."""
    from constrain.config import get_config

    cfg = get_config()
    policy = RandomizedCausalPolicy(
        risk_threshold=cfg.learned_policy_threshold,
        seed=42,
    )

    # Sample features (from your actual feature schema)
    test_features = {
        "iteration": 3.0,
        "temperature": 0.7,
        "ascii_ratio": 0.95,
        "foreign_char_ratio": 0.02,
        "repetition_score": 0.1,
        "total_energy": 0.5,
        "grounding_energy": 0.3,
        "stability_energy": 0.2,
        "is_intervention": 0,
    }

    print("Testing RandomizedCausalPolicy")
    print("=" * 60)

    for i in range(10):
        action, meta = policy.decide(test_features, problem_seed=i)
        print(f"  Problem {i}: risk={meta['risk_score']:.3f}, "
              f"action={action}, type={meta['decision_type']}")

    print("=" * 60)


if __name__ == "__main__":
    main()