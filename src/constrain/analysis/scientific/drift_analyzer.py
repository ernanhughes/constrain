from typing import Any, Dict

import numpy as np
import pandas as pd

from .base import ScientificAnalyzer


class DriftAnalyzer(ScientificAnalyzer):

    def analyze(self, df: pd.DataFrame) -> Dict[str, Any]:

        if "total_energy" not in df.columns:
            return {}

        drift_by_problem = (
            df.sort_values(["problem_id", "iteration"])
              .groupby("problem_id")["total_energy"]
              .diff()
        )

        return {
            "mean_drift": float(np.nanmean(drift_by_problem)),
            "std_drift": float(np.nanstd(drift_by_problem)),
        }