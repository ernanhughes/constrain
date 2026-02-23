# analysis/causal/propensity_model.py

from __future__ import annotations
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier


class PropensityModel:
    """
    Learns e(x) = P(T=1 | X)
    """

    def fit_predict(
        self,
        X_train: np.ndarray,
        T_train: np.ndarray,
        X_test: np.ndarray,
    ) -> np.ndarray:

        clf = GradientBoostingClassifier()
        clf.fit(X_train, T_train)

        return clf.predict_proba(X_test)[:, 1]