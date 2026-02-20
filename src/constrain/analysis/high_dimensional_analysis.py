from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
from constrain.analysis.metrics_aggregator import MetricsAggregator
from constrain.data.memory import Memory


class HighDimensionalAnalyzer:

    @staticmethod
    def run(memory: Memory, run_id: str):

        print("🔎 Building dataframe...")
        df = MetricsAggregator.build_run_dataframe(memory, run_id)

        df = df.sort_values(["problem_id", "iteration"])

        # -------------------------------------------------
        # Create future escalation target
        # -------------------------------------------------

        df["phase_next"] = df.groupby("problem_id")["phase_value"].shift(-1)
        df["escalation"] = (df["phase_next"] > df["phase_value"]).astype(int)

        for base in [
            "total_energy",
            "grounding_energy",
            "stability_energy",
            "accuracy",
            "correctness",
        ]:
            if f"{base}_y" in df.columns:
                df[base] = df[f"{base}_y"]

        # Drop all _x/_y duplicates
        df = df.drop(columns=[c for c in df.columns if c.endswith("_x") or c.endswith("_y")])

        # -----------------------------
        # Target
        # -----------------------------

        df = df.dropna()

        print("Row count after target creation:", len(df))
        print("Escalation value counts:")
        print(df["escalation"].value_counts())
        print("Phase value counts:")
        print(df["phase_value"].value_counts())


        # -------------------------------------------------
        # Drop leakage & non-numeric columns
        # -------------------------------------------------

        drop_cols = [
            "step_id",
            "run_id",
            "problem_id",
            "policy_action",
            "phase",
            "phase_next",
            "escalation",
            "correctness",
            "accuracy",
            "extracted_answer",   # ← Need to remove this because it's a direct leakage from the LLM output
            "phase_value",
        ]

        X = df.drop(columns=drop_cols, errors="ignore")
        y = df["escalation"]

        print("\n--- FEATURE COLUMNS ---")
        for col in X.columns:
            print(col)
        print("\n-----")


        # Keep only numeric columns
        X = X.select_dtypes(include=[np.number])

        print(f"Initial feature count: {X.shape[1]}")

        # -------------------------------------------------
        # Remove constant columns
        # -------------------------------------------------

        nunique = X.nunique()
        constant_cols = nunique[nunique <= 1].index
        X = X.drop(columns=constant_cols)

        print(f"After removing constant cols: {X.shape[1]}")

        # -------------------------------------------------
        # Remove near-constant columns
        # -------------------------------------------------

        low_var_cols = X.var()[X.var() < 1e-6].index
        X = X.drop(columns=low_var_cols)

        print(f"After removing low variance cols: {X.shape[1]}")

        # -------------------------------------------------
        # Group split by problem_id
        # -------------------------------------------------

        groups = df["problem_id"]

        splitter = GroupShuffleSplit(test_size=0.3, random_state=42)
        train_idx, test_idx = next(splitter.split(X, y, groups))

        X_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx]

        X_test = X.iloc[test_idx]
        y_test = y.iloc[test_idx]

        # -------------------------------------------------
        # Train XGBoost
        # -------------------------------------------------

        print("🚀 Training XGBoost...")

        model = XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method="hist",
        )

        model.fit(X_train, y_train)

        preds = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, preds)

        print(f"\n🎯 AUC: {auc:.4f}")

        # -------------------------------------------------
        # Feature Importance
        # -------------------------------------------------

        importance = pd.DataFrame({
            "feature": X.columns,
            "importance": model.feature_importances_,
        })

        importance = importance.sort_values("importance", ascending=False)

        print("\n🔥 Top 20 Predictive Signals:")
        print(importance.head(20))

        return {
            "auc": auc,
            "top_features": importance.head(20),
            "model": model,
        }
