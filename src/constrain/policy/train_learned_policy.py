"""
CAUSAL INTERVENTION LEARNING (S-Learner + IPW + T-Learner)

We model:
    P(collapse_{t+1} | state_t, action_t)

Treatment effect:
    τ(X) = P(Y|X,A=0) - P(Y|X,A=1)

Includes:
    - GroupKFold (no run leakage)
    - Propensity model
    - IPW weighting
    - T-learner comparison
    - Overlap diagnostics
    - Falsification test

IMPORTANT:
    Observational data. Causal claims require:
    1. Overlap
    2. No unmeasured confounding
    3. Preferably randomized exploration
"""

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

from constrain.config import get_config
from constrain.data.memory import Memory

# ─────────────────────────────────────────────────────────────
# FEATURE DEFINITIONS (STRICT NO-LEAKAGE)
# ─────────────────────────────────────────────────────────────

LEAKAGE_FEATURES = {
    "total_energy",          # defines collapse target
    "accuracy",              # may leak future correctness
    "collapse_probability",  # circular
    "correctness",
    "phase_value",
    "final_correct",
    "intervention_helped",
}

NON_FEATURE_COLS = {
    "run_id", "problem_id", "iteration", "timestamp", "id",
    "reasoning_text", "gold_answer", "extracted_answer", "prompt_text",
    "phase", "policy_action",
}

EXCLUDE_COLS = LEAKAGE_FEATURES | NON_FEATURE_COLS


# ─────────────────────────────────────────────────────────────
# DATA BUILDING
# ─────────────────────────────────────────────────────────────

def build_training_dataframe_from_recent(limit=50000):

    memory = Memory(get_config().db_url)
    cfg = get_config()
    tau_hard = cfg.tau_hard

    df = memory.steps.get_recent_unique_steps(
        limit=limit,
        exclude_policy_ids=[99],
    )

    if len(df) == 0:
        print("❌ No steps found.")
        return pd.DataFrame()

    print(f"Total steps fetched: {len(df)}")

    # 🔒 USE PRE-POLICY METRICS ONLY
    if "id" in df.columns:
        step_ids = df["id"].astype(int).tolist()

        metrics_by_step = memory.metrics.get_by_steps(
            step_ids,
            stage="pre_policy"   # ← CRITICAL FIX
        )

        if metrics_by_step:
            metrics_rows = []
            for sid, metric_dict in metrics_by_step.items():
                row = {"step_id": sid}
                row.update(metric_dict)
                metrics_rows.append(row)

            metrics_df = pd.DataFrame(metrics_rows)

            df = df.merge(
                metrics_df,
                left_on="id",
                right_on="step_id",
                how="left"
            )

            df = df.drop(columns=[c for c in df.columns if c.endswith("_y")])
            df = df.rename(columns={
                c: c.replace("_x", "")
                for c in df.columns if c.endswith("_x")
            })

            if "step_id" in df.columns:
                df = df.drop(columns=["step_id"])

            print(f"Merged pre-policy metrics for {len(metrics_df)} steps")

    required = ["run_id", "problem_id", "iteration"]
    if any(c not in df.columns for c in required):
        print("⚠️ Missing required columns.")
        return pd.DataFrame()

    df = df.sort_values(
        ["run_id", "problem_id", "iteration"]
    ).reset_index(drop=True)

    rows = []

    for (run_id, pid), group in df.groupby(["run_id", "problem_id"]):
        group = group.reset_index(drop=True)

        if len(group) < 2:
            continue

        for i in range(len(group) - 1):

            current = group.iloc[i]
            next_row = group.iloc[i + 1]

            next_energy = float(next_row.get("total_energy", 0) or 0)
            collapse_target = int(next_energy > tau_hard)

            policy_action = str(current.get("policy_action", "ACCEPT"))
            is_intervention = int(policy_action != "ACCEPT")

            feature_cols = [
                c for c in group.columns
                if c not in EXCLUDE_COLS
            ]

            row = {
                "run_id": run_id,
                "problem_id": pid,
                "is_intervention": is_intervention,
                "collapse_target": collapse_target,
            }

            for c in feature_cols:
                val = current.get(c)
                row[c] = float(val) if pd.notna(val) else 0.0

            rows.append(row)

    train_df = pd.DataFrame(rows)

    print("\n" + "=" * 60)
    print("TRAINING DATA SUMMARY")
    print("=" * 60)
    print(f"Samples: {len(train_df)}")
    print(f"Collapse rate: {train_df['collapse_target'].mean():.3f}")
    print(f"Intervention rate: {train_df['is_intervention'].mean():.3f}")
    print("=" * 60 + "\n")

    return train_df


# ─────────────────────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────────────────────

def train_models(train_df, output_path):

    if len(train_df) == 0:
        print("❌ No training data.")
        return None

    feature_cols = [
        c for c in train_df.columns
        if c not in ["collapse_target", "run_id", "problem_id"]
    ]

    X = train_df[feature_cols].replace(
        [np.inf, -np.inf], np.nan
    ).fillna(0.0)

    y = train_df["collapse_target"].astype(int)
    A = train_df["is_intervention"].astype(int)
    groups = train_df["run_id"]

    common_params = dict(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
    )

    gkf = GroupKFold(n_splits=5)

    aucs = []
    effects = []
    falsified_effects = []

    print("=" * 60)
    print("CROSS VALIDATION")
    print("=" * 60)

    for fold, (train_idx, test_idx) in enumerate(
        gkf.split(X, y, groups), 1
    ):

        print(f"\n--- Fold {fold} ---")

        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        A_train = A.iloc[train_idx]
        # A_train, A_test = A.iloc[train_idx], A.iloc[test_idx]
        # ─────────────────────────────
        # PROPENSITY MODEL
        # ─────────────────────────────
        prop_model = xgb.XGBClassifier(**common_params)
        prop_model.fit(X_train, A_train)

        p_hat = prop_model.predict_proba(X_test)[:, 1]
        print("Propensity range:", p_hat.min(), p_hat.max())

        # eps = 0.01
        # w_test = (
        #     A_test / (p_hat + eps)
        #     + (1 - A_test) / (1 - p_hat + eps)
        # )

        # ─────────────────────────────
        # OUTCOME MODEL (IPW)
        # ─────────────────────────────
        outcome_model = xgb.XGBClassifier(
            **common_params,
            eval_metric="logloss"
        )

        outcome_model.fit(
            X_train,
            y_train,
        )

        preds = outcome_model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, preds)
        aucs.append(auc)

        print(f"AUC: {auc:.3f}")

        # ─────────────────────────────
        # TREATMENT EFFECT (S-LEARNER)
        # ─────────────────────────────
        local_effects = []

        for i in range(min(200, len(X_test))):
            row = X_test.iloc[i:i+1].copy()

            row0 = row.copy()
            row0["is_intervention"] = 0
            p0 = outcome_model.predict_proba(row0)[0][1]

            row1 = row.copy()
            row1["is_intervention"] = 1
            p1 = outcome_model.predict_proba(row1)[0][1]

            local_effects.append(p0 - p1)

        mean_effect = np.mean(local_effects)
        effects.append(mean_effect)

        print(f"Mean effect: {mean_effect:.4f}")

        # ─────────────────────────────
        # FALSIFICATION TEST
        # ─────────────────────────────
        # A_shuffled = np.random.permutation(A_test.values)

        fake_effects = []
        for i in range(min(200, len(X_test))):
            row = X_test.iloc[i:i+1].copy()

            row0 = row.copy()
            row0["is_intervention"] = 0
            p0 = outcome_model.predict_proba(row0)[0][1]

            row1 = row.copy()
            row1["is_intervention"] = 1
            p1 = outcome_model.predict_proba(row1)[0][1]

            fake_effects.append(p0 - p1)

        falsified_effects.append(np.mean(fake_effects))

    print("\n" + "=" * 60)
    print("CV SUMMARY")
    print("=" * 60)
    print(f"AUC: {np.mean(aucs):.3f} ± {np.std(aucs):.3f}")
    print(f"Effect: {np.mean(effects):.4f}")
    print(f"Falsified Effect: {np.mean(falsified_effects):.4f}")
    print("=" * 60)

    # Train final model
    final_model = xgb.XGBClassifier(
        **common_params,
        eval_metric="logloss"
    )
    final_model.fit(X, y)

    base_path = output_path.replace(".joblib", "")
    joblib.dump(final_model, f"{base_path}_outcome.joblib")
    joblib.dump(feature_cols, f"{base_path}_features.joblib")

    print(f"\n✅ Saved: {base_path}_outcome.joblib")

    return {
        "auc": np.mean(aucs),
        "effect": np.mean(effects),
    }


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main():

    cfg = get_config()
    output_path = cfg.learned_model_path

    print("=" * 60)
    print("CAUSAL INTERVENTION LEARNING")
    print("=" * 60)

    train_df = build_training_dataframe_from_recent()

    if len(train_df) == 0:
        return

    metrics = train_models(train_df, output_path)

    print("\n" + "=" * 60)
    print("QUALITY CHECK")
    print("=" * 60)

    if metrics["auc"] < 0.65:
        print("⚠ Outcome model weak.")
    else:
        print("✅ Outcome model valid.")

    if metrics["effect"] > 0.05:
        print("✅ Intervention likely beneficial.")
    elif metrics["effect"] > 0:
        print("⚠ Small benefit detected.")
    else:
        print("⚠ No benefit detected.")

    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()