# analysis/causal/dr_estimator.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
from sklearn.model_selection import KFold

from .outcome_model import OutcomeModel
from .propensity_model import PropensityModel


@dataclass
class DRResult:
    ate: float
    std_error: float
    ci_lower: float
    ci_upper: float
    n: int
    overlap_min: float
    overlap_max: float
    ess: float


class CrossFittedDREstimator:
    """
    Cross-fitted doubly robust estimator.

    Implements:
        ψ = μ1 - μ0
            + T*(Y - μ1)/e
            - (1-T)*(Y - μ0)/(1-e)

    Cross-fitting prevents overfitting bias.
    """

    def __init__(
        self,
        n_splits: int = 5,
        clip_min: float = 0.01,
        clip_max: float = 0.99,
        seed: int = 42,
    ):
        self.n_splits = n_splits
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.seed = seed

    def estimate(
        self,
        X: np.ndarray,
        T: np.ndarray,
        Y: np.ndarray,
    ) -> DRResult:

        n = len(Y)

        kf = KFold(
            n_splits=self.n_splits,
            shuffle=True,
            random_state=self.seed,
        )

        psi_values = np.zeros(n)
        propensity_values = np.zeros(n)

        for train_idx, test_idx in kf.split(X):

            X_train, X_test = X[train_idx], X[test_idx]
            T_train, T_test = T[train_idx], T[test_idx]
            Y_train, Y_test = Y[train_idx], Y[test_idx]

            # -------------------------------------------------
            # Fit nuisance models
            # -------------------------------------------------

            outcome_model = OutcomeModel()
            mu1, mu0 = outcome_model.fit_predict(
                X_train, T_train, Y_train, X_test
            )

            prop_model = PropensityModel()
            e = prop_model.fit_predict(
                X_train, T_train, X_test
            )

            e = np.clip(e, self.clip_min, self.clip_max)

            propensity_values[test_idx] = e

            # -------------------------------------------------
            # DR score
            # -------------------------------------------------

            psi = (
                mu1 - mu0
                + T_test * (Y_test - mu1) / e
                - (1 - T_test) * (Y_test - mu0) / (1 - e)
            )

            psi_values[test_idx] = psi

        ate = float(np.mean(psi_values))

        # -------------------------------------------------
        # Robust SE
        # -------------------------------------------------

        std_error = float(np.std(psi_values) / np.sqrt(n))

        ci_lower = ate - 1.96 * std_error
        ci_upper = ate + 1.96 * std_error

        # -------------------------------------------------
        # Diagnostics
        # -------------------------------------------------

        overlap_min = float(propensity_values.min())
        overlap_max = float(propensity_values.max())

        weights = (
            T / propensity_values
            + (1 - T) / (1 - propensity_values)
        )

        ess = float((np.sum(weights) ** 2) / np.sum(weights ** 2))

        return DRResult(
            ate=ate,
            std_error=std_error,
            ci_lower=float(ci_lower),
            ci_upper=float(ci_upper),
            n=n,
            overlap_min=overlap_min,
            overlap_max=overlap_max,
            ess=ess,
        )