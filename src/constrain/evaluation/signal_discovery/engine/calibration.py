# analysis/stage3/engine/calibration.py

from __future__ import annotations

from typing import Any, Dict

import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss


class ProbabilityCalibration:

    @staticmethod
    def analyze(
        y_true,
        y_probs,
        n_bins: int = 10,
    ) -> Dict[str, Any]:

        if len(np.unique(y_true)) < 2:
            return {
                "brier_score": None,
                "expected_calibration_error": None,
                "bin_true": [],
                "bin_pred": [],
            }

        prob_true, prob_pred = calibration_curve(
            y_true,
            y_probs,
            n_bins=n_bins,
            strategy="uniform",
        )

        brier = brier_score_loss(y_true, y_probs)

        ece = float(np.mean(np.abs(prob_true - prob_pred)))

        return {
            "brier_score": float(brier),
            "expected_calibration_error": ece,
            "bin_true": prob_true.tolist(),
            "bin_pred": prob_pred.tolist(),
        }