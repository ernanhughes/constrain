import pandas as pd


class SignalDriftMonitor:

    @staticmethod
    def compute_trend(leaderboard_df: pd.DataFrame):

        leaderboard_df = leaderboard_df.sort_values("run_id")

        return {
            "auc_trend": leaderboard_df["mean_auc"].diff().mean(),
            "stability_trend": leaderboard_df["cv_ratio"].diff().mean(),
        }