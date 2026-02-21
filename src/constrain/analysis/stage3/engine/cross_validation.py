# src/constrain/analysis/stage3/engine/cross_validation.py

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold


class CrossValidator:

    def __init__(self, n_splits: int = 5):
        self.n_splits = n_splits

    def run(
        self,
        model_factory,
        X,
        y,
        groups,
    ) -> Dict[str, Any]:
        """
        Runs GroupKFold cross validation.

        model_factory: callable that returns a fresh model instance
        """

        gkf = GroupKFold(n_splits=self.n_splits)

        aucs: List[float] = []
        fold_models = []
        feature_rankings: List[List[Tuple[str, float]]] = []
        prob_outputs = []

        for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups)):

            model = model_factory()

            model.fit(X.iloc[train_idx], y.iloc[train_idx])

            preds = model.predict_proba(X.iloc[test_idx])[:, 1]

            if len(np.unique(y.iloc[test_idx])) < 2:
                continue

            auc = roc_auc_score(y.iloc[test_idx], preds)

            aucs.append(float(auc))
            fold_models.append(model)

            ranking = sorted(
                zip(X.columns, model.feature_importances_),
                key=lambda x: x[1],
                reverse=True,
            )

            feature_rankings.append(ranking)
            prob_outputs.append(
                (y.iloc[test_idx].values, preds)
            )

        return {
            "aucs": aucs,
            "models": fold_models,
            "feature_rankings": feature_rankings,
            "prob_outputs": prob_outputs,
        }