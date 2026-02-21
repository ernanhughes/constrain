# analysis/stage3/engine/stability.py

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List

import numpy as np


class SignalStabilityAnalyzer:

    @staticmethod
    def auc_stability(auc_list: List[float]) -> Dict[str, Any]:

        mean_auc = float(np.mean(auc_list))
        std_auc = float(np.std(auc_list))

        cv_ratio = (
            float(std_auc / mean_auc)
            if mean_auc > 0 else None
        )

        return {
            "mean_auc": mean_auc,
            "std_auc": std_auc,
            "cv_ratio": cv_ratio,
        }

    @staticmethod
    def feature_stability(
        feature_importances_per_fold: List[List[tuple]],
        top_k: int = 10,
    ) -> Dict[str, int]:

        top_features = []

        for ranking in feature_importances_per_fold:
            top_features.extend(
                [f for f, _ in ranking[:top_k]]
            )

        counts = Counter(top_features)

        return dict(counts)