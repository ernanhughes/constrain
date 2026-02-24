from typing import List

import pandas as pd


class SignalLeaderboard:

    @staticmethod
    def build(run_results: List[dict]):

        rows = []

        for result in run_results:
            rows.append({
                "run_id": result["run_id"],
                "mean_auc": result["mean_auc"],
                "std_auc": result["std_auc"],
                "cv_ratio": result["cv_ratio"],
                "num_features": result["num_features"],
            })

        df = pd.DataFrame(rows)

        return df.sort_values("mean_auc", ascending=False)