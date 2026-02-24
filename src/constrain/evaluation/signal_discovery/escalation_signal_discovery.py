# analysis/stage3/escalation_signal_discovery.py

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from xgboost import XGBClassifier

from constrain.analysis.aggregation.metrics_aggregator import MetricsAggregator
from constrain.data.memory import Memory


class EscalationSignalDiscovery:

    def __init__(self, memory: Memory):
        self.memory = memory

    def analyze(self, run_id: str):

        df = MetricsAggregator.build_run_dataframe(self.memory, run_id)
        df = df.sort_values(["problem_id", "iteration"])

        df["phase_next"] = df.groupby("problem_id")["phase_value"].shift(-1)
        df["escalation"] = (df["phase_next"] > df["phase_value"]).astype(int)

        df = df.dropna()

        if df["escalation"].nunique() < 2:
            raise ValueError("Escalation has only one class.")

        features = self._select_features(df)
        X = df[features]
        y = df["escalation"]

        aucs = []
        gkf = GroupKFold(n_splits=5)

        for train_idx, test_idx in gkf.split(X, y, df["problem_id"]):
            model = XGBClassifier(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                tree_method="hist",
                eval_metric="logloss",
                random_state=42,
            )

            model.fit(X.iloc[train_idx], y.iloc[train_idx])

            preds = model.predict_proba(X.iloc[test_idx])[:, 1]
            auc = roc_auc_score(y.iloc[test_idx], preds)

            aucs.append(auc)

        return {
            "mean_auc": float(np.mean(aucs)),
            "std_auc": float(np.std(aucs)),
        }

    def _select_features(self, df):

        exclude = {
            "step_id",
            "run_id",
            "problem_id",
            "phase",
            "phase_next",
            "escalation",
            "correctness",
            "accuracy",
            "extracted_answer",
            "phase_value",
        }

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        return [c for c in numeric_cols if c not in exclude]