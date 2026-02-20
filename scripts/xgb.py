import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score

from constrain.config import get_config
from constrain.data.memory import Memory
from constrain.analysis.aggregation.metrics_aggregator import MetricsAggregator


# -------------------------------------------------
# Utilities
# -------------------------------------------------

def get_memory():
    config = get_config()
    return Memory(config.db_url)


def get_latest_run_id(memory):
    runs = memory.runs.list(limit=1, desc=True)
    if not runs:
        raise ValueError("No runs found.")
    return runs[0].run_id


# -------------------------------------------------
# Main
# -------------------------------------------------

def main(run_id=None):

    memory = get_memory()

    if run_id is None:
        run_id = get_latest_run_id(memory)

    print(f"\n🔎 Building dataframe for run: {run_id}")
    df = MetricsAggregator.build_run_dataframe(memory, run_id)

    if df.empty:
        raise ValueError("Run dataframe is empty.")

    # ---------------------------------------
    # Canonicalize duplicate columns
    # ---------------------------------------

    for base in [
        "total_energy",
        "grounding_energy",
        "stability_energy",
        "accuracy",
        "correctness",
    ]:
        if f"{base}_y" in df.columns:
            df[base] = df[f"{base}_y"]

    df = df.drop(columns=[c for c in df.columns if c.endswith("_x") or c.endswith("_y")])

    # ---------------------------------------
    # Target validation
    # ---------------------------------------

    if "correctness" not in df.columns:
        raise ValueError("correctness column not found.")

    df = df.dropna()

    print("\nTarget distribution:")
    print(df["correctness"].value_counts())

    if df["correctness"].nunique() < 2:
        raise ValueError("Only one class present. Cannot compute AUC.")

    # ---------------------------------------
    # Feature selection (leakage protection)
    # ---------------------------------------

    exclude = {
        "step_id",
        "run_id",
        "problem_id",
        "policy_action",
        "phase",
        "correctness",
        "accuracy",
        "extracted_answer",  # leakage
    }

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    features = [c for c in numeric_cols if c not in exclude]

    print(f"\nFeature count: {len(features)}")

    X = df[features]
    y = df["correctness"]

    # ---------------------------------------
    # Time-aware cross validation
    # ---------------------------------------

    tscv = TimeSeriesSplit(n_splits=5)

    aucs = []
    last_model = None

    for fold, (train_idx, test_idx) in enumerate(tscv.split(df)):

        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        if y_test.nunique() < 2:
            print(f"Fold {fold} skipped (single class).")
            continue

        model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=42,
        )

        model.fit(X_train, y_train)

        preds = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, preds)

        aucs.append(float(auc))
        last_model = model

        print(f"Fold {fold} AUC: {auc:.4f}")

    if not aucs:
        raise ValueError("No valid folds for AUC computation.")

    print("\nMean AUC:", np.mean(aucs))
    print("Std AUC:", np.std(aucs))

    # ---------------------------------------
    # Feature Importance
    # ---------------------------------------

    if last_model is not None:
        importances = pd.Series(
            last_model.feature_importances_,
            index=features
        ).sort_values(ascending=False)

        print("\nTop 10 Features:")
        print(importances.head(10))

        # Optional: save importance
        importances.head(20).to_csv(f"{run_id}_top_features.csv")

    print("\n✅ XGBoost analysis complete.")


if __name__ == "__main__":
    main()