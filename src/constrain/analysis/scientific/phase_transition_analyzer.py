from typing import Any, Dict

import pandas as pd


class PhaseTransitionAnalyzer:

    def analyze(self, df: pd.DataFrame) -> Dict[str, Any]:

        transitions = (
            df.groupby("problem_id")["phase"]
              .apply(lambda x: (x != x.shift()).sum())
        )

        return {
            "mean_phase_transitions": float(transitions.mean())
        }