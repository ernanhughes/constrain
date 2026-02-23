# constrain/policy/train_learned_policy_multihead.py
"""
CAUSAL INTERVENTION LEARNING (S-Learner)

Single outcome model: P(collapse_{t+1} | state_t, action_t)
Treatment effect: τ(X) = f(X, A=0) - f(X, A=1)

CRITICAL: This is observational data. Causal claims require:
1. Overlap (both actions taken in similar states)
2. No unmeasured confounding
3. Preferably: randomized exploration
"""

import joblib
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
from constrain.data.memory import Memory
from constrain.config import get_config

# ─────────────────────────────────────────────────────────────
# FEATURE DEFINITIONS (No Leakage)
# ─────────────────────────────────────────────────────────────
LEAKAGE_FEATURES = {
    "total_energy",       # Directly defines collapse target
    "accuracy",           # Leaks future correctness  
    "collapse_probability", # Policy output (circular)
    "correctness",        # Leaks target
    "phase_value",        # Defines collapse
    "final_correct",      # Future information
    "intervention_helped", # Problem-level leakage
}

NON_FEATURE_COLS = {
    "run_id", "problem_id", "iteration", "timestamp", "id",
    "reasoning_text", "gold_answer", "extracted_answer", "prompt_text",
    "phase", "policy_action",
}

EXCLUDE_COLS = LEAKAGE_FEATURES | NON_FEATURE_COLS


# ─────────────────────────────────────────────────────────────
# DATA BUILDING (CAUSALLY CLEAN)
# ─────────────────────────────────────────────────────────────
def build_training_dataframe_from_recent(limit=50000):
    memory = Memory(get_config().db_url)
    cfg = get_config()
    tau_hard = cfg.tau_hard

    # Fetch steps
    df = memory.steps.get_recent_unique_steps(
        limit=limit,
        exclude_policy_ids=[99],
    )

    if len(df) == 0:
        print("❌ No steps found.")
        return pd.DataFrame()

    print(f"Total steps fetched: {len(df)}")

    # Merge metrics (with duplicate handling)
    if "id" in df.columns:
        step_ids = df["id"].astype(int).tolist()
        if step_ids:
            metrics_by_step = memory.metrics.get_by_steps(step_ids, stage="post_policy")
            if metrics_by_step:
                metrics_rows = []
                for sid, metric_dict in metrics_by_step.items():
                    row = {"step_id": sid}
                    row.update(metric_dict)
                    metrics_rows.append(row)
                metrics_df = pd.DataFrame(metrics_rows)

                df = df.merge(metrics_df, left_on="id", right_on="step_id", how="left")

                # Drop duplicate columns from metrics merge
                cols_to_drop = [c for c in df.columns if c.endswith("_y")]
                df = df.drop(columns=cols_to_drop)
                rename_map = {c: c.replace("_x", "") for c in df.columns if c.endswith("_x")}
                df = df.rename(columns=rename_map)
                if "step_id" in df.columns:
                    df = df.drop(columns=["step_id"])
                print(f"Merged metrics for {len(metrics_df)} steps")

    # Sort
    required_cols = ["run_id", "problem_id", "iteration"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"⚠️  Missing columns: {missing}")
        return pd.DataFrame()

    df = df.sort_values(["run_id", "problem_id", "iteration"]).reset_index(drop=True)

    # ─────────────────────────────────────────────────────────────
    # BUILD TRAINING ROWS (SINGLE OUTCOME)
    # ─────────────────────────────────────────────────────────────
    rows = []

    for (run_id, pid), group in df.groupby(["run_id", "problem_id"]):
        group = group.reset_index(drop=True)

        if len(group) < 2:
            continue

        for i in range(len(group) - 1):
            current = group.iloc[i]
            next_row = group.iloc[i + 1]

            # ─────────────────────────────────────────────────────────
            # SINGLE TARGET: Collapse (next step energy > tau)
            # ─────────────────────────────────────────────────────────
            next_energy = float(next_row.get("total_energy", 0) or 0)
            collapse_target = int(next_energy > tau_hard)

            # ─────────────────────────────────────────────────────────
            # ACTION INDICATOR (CRITICAL for causal estimation)
            # ─────────────────────────────────────────────────────────
            policy_action = str(current.get("policy_action", "ACCEPT"))
            is_intervention = int(policy_action != "ACCEPT")

            # ─────────────────────────────────────────────────────────
            # FEATURES (Exclude Leakage)
            # ─────────────────────────────────────────────────────────
            feature_cols = [c for c in group.columns if c not in EXCLUDE_COLS]

            row = {"run_id": run_id, "problem_id": pid}

            for c in feature_cols:
                val = current.get(c)
                row[c] = float(val) if pd.notna(val) else 0.0

            # Add action indicator (THIS IS THE KEY)
            row["is_intervention"] = is_intervention

            # Add single target
            row["collapse_target"] = collapse_target

            rows.append(row)

    train_df = pd.DataFrame(rows)

    print("\n" + "=" * 60)
    print("TRAINING DATA SUMMARY (S-LEARNER)")
    print("=" * 60)
    print(f"Training samples: {len(train_df)}")

    if len(train_df) > 0:
        feature_cols = [c for c in train_df.columns if c not in 
                       ["collapse_target", "run_id", "problem_id"]]
        print(f"Feature count: {len(feature_cols)}")

        print(f"\nTarget distribution:")
        print(f"  Collapse positive rate: {train_df['collapse_target'].mean():.3f}")
        print(f"  Intervention rate: {train_df['is_intervention'].mean():.3f}")

        # ─────────────────────────────────────────────────────────
        # OVERLAP DIAGNOSTIC (CRITICAL for causal validity)
        # ─────────────────────────────────────────────────────────
        print(f"\n🔍 OVERLAP DIAGNOSTIC:")
        
        # Overall intervention rate
        overall_intervene_rate = train_df["is_intervention"].mean()
        print(f"  Overall intervention rate: {overall_intervene_rate:.1%}")

        # Intervention rate by collapse outcome
        collapsed = train_df[train_df["collapse_target"] == 1]
        not_collapsed = train_df[train_df["collapse_target"] == 0]
        
        if len(collapsed) > 0:
            intervene_rate_collapsed = collapsed["is_intervention"].mean()
            print(f"  Intervention rate (collapsed): {intervene_rate_collapsed:.1%}")
        
        if len(not_collapsed) > 0:
            intervene_rate_not_collapsed = not_collapsed["is_intervention"].mean()
            print(f"  Intervention rate (not collapsed): {intervene_rate_not_collapsed:.1%}")

        # Check for extreme imbalance
        if overall_intervene_rate < 0.05 or overall_intervene_rate > 0.95:
            print(f"  ⚠️  WARNING: Extreme action imbalance — causal effect unreliable")
        elif abs(intervene_rate_collapsed - intervene_rate_not_collapsed) > 0.3:
            print(f"  ⚠️  WARNING: Action strongly confounded with outcome")
        else:
            print(f"  ✅ Reasonable overlap — causal estimation possible")

    print("=" * 60 + "\n")

    return train_df


# ─────────────────────────────────────────────────────────────
# TRAINING (S-LEARNER)
# ─────────────────────────────────────────────────────────────
def train_models(train_df, output_path):
    if len(train_df) == 0:
        print("❌ No training data.")
        return None

    feature_cols = [
        c for c in train_df.columns
        if c not in ["collapse_target", "run_id", "problem_id"]
    ]

    X = train_df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y = train_df["collapse_target"].astype(int)
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

    cv_aucs = []
    cv_effects = []

    print("\n" + "=" * 60)
    print("CROSS-VALIDATION (S-LEARNER)")
    print("=" * 60)

    for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups), 1):
        print(f"\n--- Fold {fold} ---")

        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # Train single outcome model
        outcome_model = xgb.XGBClassifier(**common_params, eval_metric="logloss")
        outcome_model.fit(X_train, y_train)

        # Evaluate
        preds = outcome_model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, preds)
        cv_aucs.append(auc)
        print(f"Outcome Model AUC: {auc:.3f}")

        # Estimate treatment effect on test set
        effects = []
        for idx in range(min(100, len(X_test))):
            row = X_test.iloc[idx:idx+1].copy()

            # Predict with ACCEPT (A=0)
            row_accept = row.copy()
            row_accept["is_intervention"] = 0
            p_accept = outcome_model.predict_proba(row_accept)[0][1]

            # Predict with INTERVENE (A=1)
            row_intervene = row.copy()
            row_intervene["is_intervention"] = 1
            p_intervene = outcome_model.predict_proba(row_intervene)[0][1]

            effect = p_accept - p_intervene
            effects.append(effect)

        mean_effect = np.mean(effects)
        cv_effects.append(mean_effect)
        print(f"Mean Treatment Effect: {mean_effect:.3f}")

    print("\n" + "=" * 60)
    print("CV SUMMARY")
    print("=" * 60)
    print(f"Outcome AUC:       {np.mean(cv_aucs):.3f} ± {np.std(cv_aucs):.3f}")
    print(f"Treatment Effect:  {np.mean(cv_effects):.3f} ± {np.std(cv_effects):.3f}")

    # ─────────────────────────────────────────────────────────
    # TRAIN FINAL MODEL ON ALL DATA
    # ─────────────────────────────────────────────────────────
    outcome_model = xgb.XGBClassifier(**common_params, eval_metric="logloss")
    outcome_model.fit(X, y)

    # ─────────────────────────────────────────────────────────
    # SAVE MODEL
    # ─────────────────────────────────────────────────────────
    base_path = output_path.replace(".joblib", "")
    joblib.dump(outcome_model, f"{base_path}_outcome.joblib")
    joblib.dump(feature_cols, f"{base_path}_features.joblib")

    print(f"\n✅ Model saved: {base_path}_outcome.joblib")

    # ─────────────────────────────────────────────────────────
    # TREATMENT EFFECT ANALYSIS
    # ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("TREATMENT EFFECT ANALYSIS")
    print("=" * 60)

    # Sample high-risk states
    risk_preds = outcome_model.predict_proba(X)[:, 1]
    high_risk_idx = np.where(risk_preds > 0.7)[0][:20]

    if len(high_risk_idx) > 0:
        effects = []
        for idx in high_risk_idx:
            row = X.iloc[idx:idx+1].copy()

            row_accept = row.copy()
            row_accept["is_intervention"] = 0
            p_accept = outcome_model.predict_proba(row_accept)[0][1]

            row_intervene = row.copy()
            row_intervene["is_intervention"] = 1
            p_intervene = outcome_model.predict_proba(row_intervene)[0][1]

            effect = p_accept - p_intervene
            effects.append(effect)

        mean_effect = np.mean(effects)
        print(f"Mean effect (high-risk): {mean_effect:.3f}")

        positive_effects = sum(1 for e in effects if e > 0)
        print(f"Positive effects: {positive_effects}/{len(effects)} ({positive_effects/len(effects):.0%})")

        if mean_effect > 0.05:
            print("  ✅ Intervention appears beneficial")
        elif mean_effect > 0:
            print("  ⚠️  Small benefit detected")
        else:
            print("  ⚠️  Intervention may be ineffective or harmful")

    print("=" * 60 + "\n")

    return {
        "outcome_auc": np.mean(cv_aucs),
        "treatment_effect": np.mean(cv_effects),
    }


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
def main():
    cfg = get_config()
    output_path = cfg.learned_model_path

    print("=" * 60)
    print("CAUSAL INTERVENTION LEARNING (S-LEARNER)")
    print("=" * 60)

    train_df = build_training_dataframe_from_recent(limit=50000)

    if len(train_df) == 0:
        print("❌ Training failed: No data")
        return

    metrics = train_models(train_df, output_path)

    # Quality check
    print("\n" + "=" * 60)
    print("QUALITY CHECK")
    print("=" * 60)

    if metrics["outcome_auc"] < 0.65:
        print("⚠️  Outcome model weak — improve features or data")
    else:
        print("✅ Outcome model valid")

    if metrics["treatment_effect"] > 0.05:
        print("✅ Intervention shows positive effect")
    elif metrics["treatment_effect"] > 0:
        print("⚠️  Small effect — consider randomized experiment")
    else:
        print("⚠️  No effect detected — intervention may need redesign")

    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()