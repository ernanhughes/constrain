# constrain/policy/train_learned_policy_multihead.py
import joblib
import pandas as pd
import numpy as np
import xgboost as xgb

from constrain.data.memory import Memory
from constrain.config import get_config


# constrain/policy/train_learned_policy_multihead.py
# constrain/policy/train_learned_policy_multihead.py
def build_training_dataframe_from_recent(limit=50000):
    memory = Memory(get_config().db_url)
    cfg = get_config()
    tau_hard = cfg.tau_hard

    # ─────────────────────────────────────────────────────────────
    # 1. Fetch recent steps
    # ─────────────────────────────────────────────────────────────
    df = memory.steps.get_recent_unique_steps(limit=limit, exclude_policy_ids=[99])
    print("Total recent steps fetched:", len(df))
    print("Columns:", list(df.columns))

    if len(df) == 0:
        print("❌ No steps found.")
        return pd.DataFrame()

    # ─────────────────────────────────────────────────────────────
    # 2. Fetch problem_summaries for utility labels
    # ─────────────────────────────────────────────────────────────
    try:
        run_ids = df["run_id"].unique().tolist()
        summaries = memory.problem_summaries.get_by_run_ids(run_ids)
        
        if summaries:
            summaries_df = pd.DataFrame([{
                "run_id": s.run_id,
                "problem_id": s.problem_id,
                "intervention_helped": s.intervention_helped,
                "final_correct": s.final_correct,
            } for s in summaries])
            df = df.merge(summaries_df, on=["run_id", "problem_id"], how="left")
            print(f"Merged {len(summaries_df)} problem summaries")
    except Exception as e:
        print(f"⚠️  problem_summaries query failed: {e}")

    # ─────────────────────────────────────────────────────────────
    # 3. ⭐ CRITICAL: Fetch metrics from metrics table ⭐
    # ─────────────────────────────────────────────────────────────
    if "id" in df.columns:
        step_ids = df["id"].astype(int).tolist()
        
        if step_ids:
            metrics_by_step = memory.metrics.get_by_steps(step_ids, stage="post_policy")
            
            if metrics_by_step:
                metrics_rows = []
                for step_id, metric_dict in metrics_by_step.items():
                    row = {"step_id": step_id}
                    row.update(metric_dict)
                    metrics_rows.append(row)
                
                metrics_df = pd.DataFrame(metrics_rows)
                
                # ⭐ Identify overlapping columns (exclude step_id)
                overlap_cols = set(df.columns) & set(metrics_df.columns) - {"step_id", "id"}
                print(f"Overlapping columns (will drop _y versions): {overlap_cols}")
                
                # Merge
                df = df.merge(metrics_df, left_on="id", right_on="step_id", how="left")
                
                # ⭐ FIX: Drop _y columns (metrics duplicates), keep _x (steps originals)
                cols_to_drop = [c for c in df.columns if c.endswith("_y")]
                df = df.drop(columns=cols_to_drop)
                
                # ⭐ Rename _x columns back to original names
                rename_map = {c: c.replace("_x", "") for c in df.columns if c.endswith("_x")}
                df = df.rename(columns=rename_map)
                
                df = df.drop(columns=["step_id"])
                print(f"Merged metrics for {len(metrics_df)} steps")
            else:
                print("⚠️  No metrics found for steps — check stage parameter")
    else:
        print("⚠️  'id' column missing — skipping metrics join")

    # ─────────────────────────────────────────────────────────────
    # 4. Verify required columns exist before sorting
    # ─────────────────────────────────────────────────────────────
    required_cols = ["run_id", "problem_id", "iteration"]
    missing = [c for c in required_cols if c not in df.columns]
    
    if missing:
        print(f"⚠️  ERROR: Missing required columns: {missing}")
        print("   Available columns:", list(df.columns))
        return pd.DataFrame()

    # ─────────────────────────────────────────────────────────────
    # 5. Sort for trajectory building
    # ─────────────────────────────────────────────────────────────
    df = df.sort_values(["run_id", "problem_id", "iteration"]).reset_index(drop=True)

    # ─────────────────────────────────────────────────────────────
    # 6. Build training rows
    # ─────────────────────────────────────────────────────────────
    rows = []
    for (run_id, pid), group in df.groupby(["run_id", "problem_id"]):
        group = group.reset_index(drop=True)

        if len(group) < 2:
            continue

        problem_summary = group.iloc[0]
        
        # ⭐ FIX: Handle NaN safely with pd.notna() check
        final_correct_raw = problem_summary.get("final_correct", 0)
        if pd.notna(final_correct_raw):
            final_correct = int(final_correct_raw)
        else:
            # Fallback: use last step accuracy
            last_accuracy = group.iloc[-1].get("accuracy", 0)
            final_correct = int(float(last_accuracy or 0) > 0.5)
        
        intervention_helped_raw = problem_summary.get("intervention_helped", 0)
        if pd.notna(intervention_helped_raw):
            intervention_helped = int(intervention_helped_raw)
        else:
            # Fallback: assume no intervention helped if no summary
            intervention_helped = 0

        for i in range(len(group) - 1):
            current = group.iloc[i]
            next_row = group.iloc[i + 1]

            # HEAD 1: Collapse
            collapse_next = int(float(next_row.get("total_energy", 0)) > tau_hard)

            # HEAD 2: Utility
            intervened = str(current.get("policy_action", "ACCEPT")) != "ACCEPT"
            utility_target = int(intervened and intervention_helped)

            # HEAD 3: Delta
            current_correct_raw = current.get("accuracy", 0)
            current_correct = int(float(current_correct_raw or 0) > 0.5)
            delta_target = float(final_correct - current_correct)

            # FEATURES
            exclude_cols = {
                "run_id", "problem_id", "iteration", "timestamp", "id",
                "reasoning_text", "gold_answer", "extracted_answer", "prompt_text",
                "phase", "policy_action", "phase_value",
                "intervention_helped", "final_correct",
                "collapse_target", "utility_target", "delta_target",
            }
            
            feature_cols = [c for c in group.columns if c not in exclude_cols]

            row = {}
            for c in feature_cols:
                val = current.get(c)
                if pd.isna(val):
                    row[c] = 0.0
                else:
                    try:
                        row[c] = float(val)
                    except (ValueError, TypeError):
                        row[c] = 0.0

            if len(row) < 5:
                continue

            row["collapse_target"] = collapse_next
            row["utility_target"] = utility_target
            row["delta_target"] = delta_target

            rows.append(row)

    train_df = pd.DataFrame(rows)

    print("\n" + "="*60)
    print("TRAINING DATA SUMMARY")
    print("="*60)
    print(f"Training samples: {len(train_df)}")
    
    if len(train_df) > 0:
        feature_cols = [c for c in train_df.columns if c not in ['collapse_target', 'utility_target', 'delta_target']]
        print(f"Feature count: {len(feature_cols)}")
        print(f"\nTarget distributions:")
        print(f"  Collapse positive rate: {train_df['collapse_target'].mean():.3f}")
        print(f"  Utility positive rate: {train_df['utility_target'].mean():.3f}")
        print(f"  Delta mean/std: {train_df['delta_target'].mean():.3f} / {train_df['delta_target'].std():.3f}")
        
        print(f"\nTop 10 features by coverage:")
        coverage = [(c, train_df[c].notna().mean()) for c in feature_cols]
        coverage.sort(key=lambda x: -x[1])
        for c, cov in coverage[:10]:
            if cov > 0.3:
                print(f"  {c}: {cov:.1%}")
    print("="*60 + "\n")

    return train_df

def coerce_numeric(df: pd.DataFrame, exclude_cols: set) -> pd.DataFrame:
    """Safely coerce all non-excluded columns to numeric."""
    df = df.copy()
    for c in df.columns:
        if c in exclude_cols:
            continue
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return df


def train_models(train_df, output_path):
    """Train 3 separate models (XGBoost doesn't support multi-head natively)."""
    
    if len(train_df) == 0:
        print("❌ No training data. Aborting.")
        return

    # Prepare features
    target_cols = {"collapse_target", "utility_target", "delta_target"}
    feature_cols = [c for c in train_df.columns if c not in target_cols]
    
    X = coerce_numeric(train_df[feature_cols], exclude_cols=set())
    
    y_collapse = train_df["collapse_target"].astype(int)
    y_utility = train_df["utility_target"].astype(int)
    y_delta = train_df["delta_target"].astype(float)

    # ─────────────────────────────────────────────────────────────
    # Model configurations
    # ─────────────────────────────────────────────────────────────
    common_params = {
        "n_estimators": 300,
        "max_depth": 5,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
    }

    collapse_model = xgb.XGBClassifier(
        **common_params,
        eval_metric="logloss",
    )

    utility_model = xgb.XGBClassifier(
        **common_params,
        eval_metric="logloss",
    )

    delta_model = xgb.XGBRegressor(
        **common_params,
    )

    # ─────────────────────────────────────────────────────────────
    # Train
    # ─────────────────────────────────────────────────────────────
    print("Training Collapse Head...")
    collapse_model.fit(X, y_collapse)
    collapse_preds = collapse_model.predict_proba(X)[:, 1]
    print(f"  Collapse AUC (train): {roc_auc_score(y_collapse, collapse_preds):.3f}")

    print("Training Utility Head...")
    utility_model.fit(X, y_utility)
    utility_preds = utility_model.predict_proba(X)[:, 1]
    print(f"  Utility AUC (train): {roc_auc_score(y_utility, utility_preds):.3f}")

    print("Training Delta Head...")
    delta_model.fit(X, y_delta)
    delta_preds = delta_model.predict(X)
    delta_corr = np.corrcoef(delta_preds, y_delta)[0, 1]
    print(f"  Delta Correlation (train): {delta_corr:.3f}")

    # ─────────────────────────────────────────────────────────────
    # Save
    # ─────────────────────────────────────────────────────────────
    # Remove .joblib extension if present (we'll add head-specific suffixes)
    base_path = output_path.replace(".joblib", "")

    joblib.dump(collapse_model, f"{base_path}_collapse.joblib")
    joblib.dump(utility_model, f"{base_path}_utility.joblib")
    joblib.dump(delta_model, f"{base_path}_delta.joblib")

    print("\n" + "="*60)
    print("MODELS SAVED")
    print("="*60)
    print(f"  {base_path}_collapse.joblib")
    print(f"  {base_path}_utility.joblib")
    print(f"  {base_path}_delta.joblib")
    print("="*60 + "\n")

    return {
        "collapse_auc": roc_auc_score(y_collapse, collapse_preds),
        "utility_auc": roc_auc_score(y_utility, utility_preds),
        "delta_corr": delta_corr,
        "feature_count": X.shape[1],
    }


def main():
    cfg = get_config()
    output_path = cfg.learned_model_path

    print("="*60)
    print("3-HEAD POLICY TRAINING")
    print("="*60)

    # Build training data
    train_df = build_training_dataframe_from_recent(limit=50000)

    if len(train_df) == 0:
        print("❌ Training failed: No data")
        return

    # Train models
    metrics = train_models(train_df, output_path)

    # Quick quality check
    print("\n" + "="*60)
    print("QUALITY CHECK")
    print("="*60)
    if metrics["collapse_auc"] < 0.65:
        print("⚠️  WARNING: Collapse AUC < 0.65 — head may not be predictive")
    else:
        print("✅ Collapse head looks good")

    if metrics["utility_auc"] < 0.55:
        print("⚠️  WARNING: Utility AUC < 0.55 — may need more intervention data")
    else:
        print("✅ Utility head looks good")

    if abs(metrics["delta_corr"]) < 0.15:
        print("⚠️  WARNING: Delta correlation < 0.15 — may need redefinition")
    else:
        print("✅ Delta head looks good")

    print("="*60 + "\n")


if __name__ == "__main__":
    from sklearn.metrics import roc_auc_score
    main()