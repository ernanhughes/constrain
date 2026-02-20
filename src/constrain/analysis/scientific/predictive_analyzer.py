import pandas as pd
from typing import Dict, Any
from sklearn.metrics import roc_auc_score


class PredictiveAnalyzer:

    def analyze(self, df: pd.DataFrame) -> Dict[str, Any]:

        if "collapse_label" not in df.columns:
            return {}

        if df["collapse_label"].nunique() < 2:
            return {"predictive_auc": None}

        auc = roc_auc_score(
            df["collapse_label"],
            df["total_energy"]
        )

        return {
            "predictive_auc": float(auc)
        }