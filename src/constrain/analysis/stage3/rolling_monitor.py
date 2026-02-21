from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


class RollingSignalMonitor:
    """
    Monitors drift and instability of signals over time within a run.

    Designed to detect:
        - Gradual degradation
        - Energy escalation
        - Accuracy decay
        - Signal instability
    """

    def __init__(
        self,
        window: int = 20,
        slope_threshold: float = 0.001,
        volatility_threshold: float = 0.05,
    ):
        self.window = window
        self.slope_threshold = slope_threshold
        self.volatility_threshold = volatility_threshold

    # ---------------------------------------------------------
    # Public API
    # ---------------------------------------------------------

    def analyze_run(
        self,
        df: pd.DataFrame,
        signal_columns: Optional[List[str]] = None,
    ) -> Dict[str, Any]:

        if "iteration" not in df.columns:
            raise ValueError("DataFrame must contain 'iteration' column.")

        df = df.sort_values(["problem_id", "iteration"])

        if signal_columns is None:
            signal_columns = self._infer_numeric_signals(df)

        results = {}

        for col in signal_columns:
            results[col] = self._analyze_signal(df, col)

        return results

    # ---------------------------------------------------------
    # Signal Analysis
    # ---------------------------------------------------------

    def _analyze_signal(
        self,
        df: pd.DataFrame,
        column: str,
    ) -> Dict[str, Any]:

        if column not in df.columns:
            return {"error": "column_not_found"}

        values = df[column].dropna()

        if len(values) < self.window:
            return {"error": "not_enough_data"}

        rolling_mean = values.rolling(self.window).mean()
        rolling_std = values.rolling(self.window).std()

        slope = self._compute_slope(rolling_mean)

        volatility = rolling_std.mean()

        drift_flag = abs(slope) > self.slope_threshold
        instability_flag = volatility > self.volatility_threshold

        return {
            "mean": float(values.mean()),
            "std": float(values.std()),
            "rolling_mean_last": float(rolling_mean.iloc[-1]),
            "rolling_std_last": float(rolling_std.iloc[-1]),
            "slope": float(slope),
            "volatility": float(volatility),
            "drift_detected": bool(drift_flag),
            "instability_detected": bool(instability_flag),
        }

    # ---------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------

    def _compute_slope(self, series: pd.Series) -> float:
        series = series.dropna()

        if len(series) < 2:
            return 0.0

        x = np.arange(len(series))
        y = series.values

        # Linear regression slope
        slope = np.polyfit(x, y, 1)[0]
        return float(slope)

    def _infer_numeric_signals(self, df: pd.DataFrame) -> List[str]:
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        excluded = {
            "step_id",
            "problem_id",
            "iteration",
        }

        return [c for c in numeric_cols if c not in excluded]