# analysis/causal/outcome_model.py

from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor


@dataclass
class OutcomeModel:
    """
    Learns:
        μ1(x) = E[Y | T=1, X]
        μ0(x) = E[Y | T=0, X]
    """

    def fit_predict(
        self,
        X_train: np.ndarray,
        T_train: np.ndarray,
        Y_train: np.ndarray,
        X_test: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:

        model_treated = GradientBoostingRegressor()
        model_control = GradientBoostingRegressor()

        # Train on treated
        model_treated.fit(
            X_train[T_train == 1],
            Y_train[T_train == 1],
        )

        # Train on control
        model_control.fit(
            X_train[T_train == 0],
            Y_train[T_train == 0],
        )

        mu1 = model_treated.predict(X_test)
        mu0 = model_control.predict(X_test)

        return mu1, mu0