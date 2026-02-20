import pandas as pd
from typing import Dict, Any



class PhaseTransitionAnalyzer:

    def analyze(self, df: pd.DataFrame) -> Dict[str, Any]:

        transitions = (
            df.groupby("problem_id")["phase"]
              .apply(lambda x: (x != x.shift()).sum())
        )

        return {
            "mean_phase_transitions": float(transitions.mean())
        }