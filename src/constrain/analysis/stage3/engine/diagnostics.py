# src/constrain/analysis/stage3/engine/diagnostics.py

from __future__ import annotations
from typing import Dict, Any, List, Tuple

import numpy as np
from collections import Counter

from .calibration import ProbabilityCalibration


class SignalDiagnostics:

    @staticmethod
    def auc_stats(aucs: List[float]) -> Dict[str, Any]:

        if not aucs:
            return {
                "mean_auc": None,
                "std_auc": None,
                "cv_ratio": None,
            }

        mean_auc = float(np.mean(aucs))
        std_auc = float(np.std(aucs))
        cv_ratio = float(std_auc / mean_auc) if mean_auc > 0 else None

        return {
            "mean_auc": mean_auc,
            "std_auc": std_auc,
            "cv_ratio": cv_ratio,
        }

    @staticmethod
    def feature_stability(
        feature_rankings: List[List[Tuple[str, float]]],
        top_k: int = 10,
    ) -> Dict[str, int]:

        top_features = []

        for ranking in feature_rankings:
            top_features.extend(
                [f for f, _ in ranking[:top_k]]
            )

        counts = Counter(top_features)

        return dict(counts)

    @staticmethod
    def calibration_stats(prob_outputs):

        calibration_results = []

        for y_true, probs in prob_outputs:
            stats = ProbabilityCalibration.analyze(y_true, probs)
            calibration_results.append(stats)

        return calibration_results