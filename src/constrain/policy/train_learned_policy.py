# constrain/policy/train_learned_policy_multihead.py

import joblib
import pandas as pd
import numpy as np
import xgboost as xgb

from constrain.data.memory import Memory
from constrain.config import get_config

from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score


# ============================================================
# DATASET BUILDING
# ============================================================

def build_training_dataframe_from_recent(limit=50000):
    memory = Memory(get_config().db_url)

    df = memory.steps.get_recent_unique_steps(
        limit=limit,
        exclude_policy_ids=[99],
    )

    if len(df) == 0:
        print("❌ No steps found.")
        return pd.DataFrame()

    # Merge problem summaries (for intervention labels)
    run_ids = df["run_id"].unique().tolist()
    summaries = memory.problem_summaries.get_by_run_ids(run_ids)

    if summaries:
        summaries_df = pd.DataFrame([
            {
                "run_id": s.run_id,
                "problem_id": s.problem_id,
                "intervention_helped": s.intervention_helped,
            }
            for s in summaries
        ])
        df = df.merge(summaries_df, on=["run_id", "problem_id"], how="left")

    # Merge metrics
    step_ids = df["id"].astype(int).tolist()
    metrics_by_step = memory.metrics.get_by_steps(step_ids, stage="post_policy")

    if metrics_by_step:
        rows = []
        for sid, metric_dict in metrics_by_step.items():
            row = {"step_id": sid}
            row.update(metric_dict)
            rows.append(row)

        metrics_df = pd.DataFrame(rows)

        df = df.merge(metrics_df, left_on="id", right_on="step_id", how="left")
        df = df.drop(columns=["step_id"])

    # ------------------------------------------------------------
    # FIX column collisions from metrics merge
    # ------------------------------------------------------------

    if "iteration_x" in df.columns:
        df = df.rename(columns={"iteration_x": "iteration"})

    if "iteration_y" in df.columns:
        df = df.drop(columns=["iteration_y"])

    if "accuracy_x" in df.columns:
        df = df.rename(columns={"accuracy_x": "accuracy"})

    if "accuracy_y" in df.columns:
        df = df.drop(columns=["accuracy_y"])

    if "total_energy_x" in df.columns:
        df = df.rename(columns={"total_energy_x": "total_energy"})

    if "total_energy_y" in df.columns:
        df = df.drop(columns=["total_energy_y"])

    if "extracted_answer_x" in df.columns:
        df = df.rename(columns={"extracted_answer_x": "extracted_answer"})

    if "extracted_answer_y" in df.columns:
        df = df.drop(columns=["extracted_answer_y"])
        
    # Ensure phase_value exists
    if "phase_value" not in df.columns:
        raise ValueError("phase_value missing — collapse target invalid")

    required_cols = ["run_id", "problem_id", "iteration"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns after merge: {missing}")

    print("\nColumns before sort:")
    print(df.columns.tolist())

    df = df.sort_values(["run_id", "problem_id", "iteration"]).reset_index(drop=True)

    rows = []

    for (run_id, pid), group in df.groupby(["run_id", "problem_id"]):
        group = group.reset_index(drop=True)

        if len(group) < 2:
            continue

        raw_val = group.iloc[0].get("intervention_helped")

        if pd.notna(raw_val):
            intervention_helped = int(raw_val)
        else:
            intervention_helped = 0

        for i in range(len(group) - 1):
            current = group.iloc[i]
            next_row = group.iloc[i + 1]

            # Collapse target
            def persistent_collapse(phases, k=2):
                for j in range(len(phases) - k + 1):
                    if all(p >= 3 for p in phases[j:j+k]):
                        return True
                return False

            future_window = group.iloc[i+1:i+1+3]  # look ahead max 3 steps
            future_phases = future_window["phase_value"].fillna(0).astype(float).tolist()

            collapse_target = int(persistent_collapse(future_phases, k=2))

            # Utility target (step-level improvement)
            curr_acc = float(current.get("accuracy", 0) or 0)
            next_acc = float(next_row.get("accuracy", 0) or 0)
            utility_target = int(next_acc > curr_acc)

            # Delta target (local only)
            delta_target = next_acc - curr_acc

            exclude_cols = {
                "run_id", "problem_id", "iteration", "timestamp", "id",
                "reasoning_text", "gold_answer", "extracted_answer",
                "prompt_text", "phase", "policy_action",
                "phase_value", "intervention_helped",
                "collapse_probability",
                "accuracy", "total_energy", "correctness",
            }

            feature_cols = [c for c in group.columns if c not in exclude_cols]

            row = {"run_id": run_id, "problem_id": pid}

            for c in feature_cols:
                val = current.get(c)
                row[c] = float(val) if pd.notna(val) else 0.0

            row["collapse_target"] = collapse_target
            row["utility_target"] = utility_target
            row["delta_target"] = delta_target

            rows.append(row)

    train_df = pd.DataFrame(rows)

    print("\nTraining samples:", len(train_df))
    print("Collapse rate:", train_df["collapse_target"].mean())
    print("Utility rate:", train_df["utility_target"].mean())

    return train_df


# ============================================================
# TRAINING
# ============================================================

def train_models(train_df, output_path):

    if len(train_df) == 0:
        print("❌ No data.")
        return None

    target_cols = {"collapse_target", "utility_target", "delta_target"}

    feature_cols = [
        c for c in train_df.columns
        if c not in target_cols and c not in ["run_id", "problem_id"]
    ]

    X = train_df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    y_collapse = train_df["collapse_target"]
    y_utility = train_df["utility_target"]
    y_delta = train_df["delta_target"]

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

    collapse_aucs = []
    utility_aucs = []
    delta_corrs = []

    for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y_collapse, groups), 1):

        print(f"\n===== Fold {fold} =====")

        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]

        y_c_train = y_collapse.iloc[train_idx]
        y_c_test = y_collapse.iloc[test_idx]

        y_u_train = y_utility.iloc[train_idx]
        y_u_test = y_utility.iloc[test_idx]

        y_d_train = y_delta.iloc[train_idx]
        y_d_test = y_delta.iloc[test_idx]

        pos = y_u_train.sum()
        neg = len(y_u_train) - pos
        scale_pos_weight = (neg / pos) if pos > 0 else 1.0

        collapse_model = xgb.XGBClassifier(**common_params)
        utility_model = xgb.XGBClassifier(**common_params, scale_pos_weight=scale_pos_weight)
        delta_model = xgb.XGBRegressor(**common_params)

        collapse_model.fit(X_train, y_c_train)
        collapse_auc = roc_auc_score(
            y_c_test,
            collapse_model.predict_proba(X_test)[:, 1]
        )
        collapse_aucs.append(collapse_auc)

        utility_model.fit(X_train, y_u_train)
        utility_auc = roc_auc_score(
            y_u_test,
            utility_model.predict_proba(X_test)[:, 1]
        ) if y_u_test.nunique() > 1 else 0.5
        utility_aucs.append(utility_auc)

        delta_model.fit(X_train, y_d_train)
        delta_preds = delta_model.predict(X_test)
        delta_corr = np.corrcoef(delta_preds, y_d_test)[0, 1]
        delta_corrs.append(delta_corr)

        print(f"Collapse AUC: {collapse_auc:.3f}")
        print(f"Utility AUC: {utility_auc:.3f}")
        print(f"Delta Corr: {delta_corr:.3f}")

    print("\n=== CV Summary ===")
    print(f"Collapse AUC: {np.mean(collapse_aucs):.3f} ± {np.std(collapse_aucs):.3f}")
    print(f"Utility AUC:  {np.mean(utility_aucs):.3f} ± {np.std(utility_aucs):.3f}")
    print(f"Delta Corr:   {np.mean(delta_corrs):.3f} ± {np.std(delta_corrs):.3f}")

    # Train final models on all data
    collapse_model.fit(X, y_collapse)
    utility_model.fit(X, y_utility)
    delta_model.fit(X, y_delta)

    base_path = output_path.replace(".joblib", "")

    joblib.dump(collapse_model, f"{base_path}_collapse.joblib")
    joblib.dump(utility_model, f"{base_path}_utility.joblib")
    joblib.dump(delta_model, f"{base_path}_delta.joblib")

    print("\nModels saved.")

    return {
        "collapse_auc": np.mean(collapse_aucs),
        "utility_auc": np.mean(utility_aucs),
        "delta_corr": np.mean(delta_corrs),
    }


# ============================================================
# MAIN
# ============================================================

def main():
    cfg = get_config()
    output_path = cfg.learned_model_path

    train_df = build_training_dataframe_from_recent(limit=50000)

    if len(train_df) == 0:
        return

    train_models(train_df, output_path)


if __name__ == "__main__":
    main()