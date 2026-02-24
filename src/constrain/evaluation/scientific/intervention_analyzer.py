from typing import Any, Dict

import pandas as pd

from .base import ScientificAnalyzer


class InterventionAnalyzer(ScientificAnalyzer):

    def analyze(self, df: pd.DataFrame) -> Dict[str, Any]:

        if "policy_action" not in df.columns:
            return {}

        interventions = df[df["policy_action"] != "ACCEPT"]

        if interventions.empty:
            return {"intervention_rate": 0.0}

        return {
            "intervention_rate": float(len(interventions) / len(df)),
            "mean_energy_during_intervention": float(
                interventions["total_energy"].mean()
            )
        }