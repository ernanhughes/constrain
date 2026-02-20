# scripts/analyze_run_xgb.py

import xgboost as xgb
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from constrain.analysis.metrics_aggregator import MetricsAggregator
from constrain.data.memory import Memory

RUN_ID = "run_18cdc06e"


memory = Memory()

print("🔎 Building dataframe...")
df = MetricsAggregator.build_run_dataframe(memory, RUN_ID)

# ---------------------------------------
# Canonicalize duplicated columns
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

# Drop all _x/_y duplicates
df = df.drop(columns=[c for c in df.columns if c.endswith("_x") or c.endswith("_y")])

# -----------------------------
# Target
# -----------------------------

df = df.dropna()

print("Target distribution:")
print(df["correctness"].value_counts())

# -----------------------------
# Features
# -----------------------------

exclude = {
    "step_id",
    "run_id",
    "problem_id",
    "policy_action",
    "phase",
    "correctness",
    "accuracy",
    "extracted_answer",   # ← Need to remove this because it's a direct leakage from the LLM output
}

features = [c for c in df.columns if c not in exclude]

print("\n--- FEATURE COLUMNS ---")
for col in df.columns:
    if col not in exclude:
        print(col)
print("\n-----")

X = df[features]
y = df["correctness"]

print("Feature count:", len(features))

# -----------------------------
# Train/Test Split
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

model = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
)

model.fit(X_train, y_train)

preds = model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, preds)

print("AUC:", auc)

# -----------------------------
# Feature Importance
# -----------------------------

importances = pd.Series(
    model.feature_importances_,
    index=features
).sort_values(ascending=False)

print("Top 10 Features:")
print(importances.head(10))
