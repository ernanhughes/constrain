"""constrain.services.causal_evaluation_service
Service for evaluating causal effects of interventions in runs.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from constrain.data.memory import Memory

class DoublyRobustEstimator:

    def __init__(self, n_splits=5, seed=42):
        self.n_splits = n_splits
        self.seed = seed

    def estimate(self, df, feature_cols):

        X = df[feature_cols].values
        T = df["treatment"].values
        Y = df["collapsed"].values
        e_logged = df["propensity"].values

        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.seed)

        dr_scores = []

        for train_idx, test_idx in kf.split(X):

            # Outcome model
            mu_model = GradientBoostingRegressor()
            mu_model.fit(X[train_idx], Y[train_idx])

            mu_hat = mu_model.predict(X[test_idx])

            # Propensity model (if needed)
            e_model = LogisticRegression()
            e_model.fit(X[train_idx], T[train_idx])
            e_hat = e_model.predict_proba(X[test_idx])[:,1]

            # DR correction
            correction = (
                T[test_idx] * (Y[test_idx] - mu_hat) / e_logged[test_idx]
            )

            dr = mu_hat + correction
            dr_scores.extend(dr)

        ate = np.mean(dr_scores)
        se = np.std(dr_scores) / np.sqrt(len(dr_scores))

        return {
            "ate": float(ate),
            "ci_lower": float(ate - 1.96 * se),
            "ci_upper": float(ate + 1.96 * se),
        }
 

class CausalEvaluationService:

    def __init__(self, memory):
        self.memory = memory

    # ---------------------------------------------------------
    # Public API
    # ---------------------------------------------------------

    def evaluate_run(self, run_id: str):

        df = self._build_causal_dataframe(run_id)

        if df.empty:
            return None

        results = {}

        results["naive"] = self._naive_difference(df)
        results["ipw"] = self._ipw(df)
        results["dr"] = self._cross_fitted_dr(df)

        # Persist
        for method, r in results.items():
            self.memory.causal_evaluations.create(
                run_id=run_id,
                method=method,
                ate=r["ate"],
                ci_lower=r["ci_lower"],
                ci_upper=r["ci_upper"],
                n_samples=len(df),
            )

        return results


    def simulate_policy(self, df):

        feature_cols = [...]
        X = df[feature_cols].values

        mu_model = GradientBoostingRegressor()
        mu_model.fit(X, df["collapse_within_3"])

        # Counterfactual: no treatment
        df_cf0 = df.copy()
        df_cf0["treatment_any"] = 0

        y0 = mu_model.predict(df_cf0[feature_cols])

        # Counterfactual: always treat
        df_cf1 = df.copy()
        df_cf1["treatment_any"] = 1

        y1 = mu_model.predict(df_cf1[feature_cols])

        return {
            "always_treat_collapse_rate": float(np.mean(y1)),
            "never_treat_collapse_rate": float(np.mean(y0)),
        }

    # ---------------------------------------------------------
    # Build Causal Dataset
    # ---------------------------------------------------------

    def _build_causal_dataframe(self, run_id):

        steps = self.memory.steps.get_by_run(run_id)
        snapshots = self.memory.reasoning_state_snapshots.get_by_run(run_id)

        if not steps or not snapshots:
            return pd.DataFrame()

        steps_df = pd.DataFrame([s.model_dump() for s in steps])
        snap_df = pd.DataFrame([s.model_dump() for s in snapshots])

        df = steps_df.merge(
            snap_df,
            on=["run_id", "problem_id", "iteration"],
            how="inner",
        )

        # ---------------------------------------------------------
        # Treatment Encoding (multi-class → one-hot)
        # ---------------------------------------------------------
        df["treatment_any"] = (df["policy_action"] != "ACCEPT").astype(int)
        df["treatment_rollback"] = (df["policy_action"] == "ROLLBACK").astype(int)
        df["treatment_reset"] = (df["policy_action"] == "RESET").astype(int)

        # Temperature shift
        df["delta_temp"] = df["temperature"] - df.groupby("problem_id")["temperature"].shift(1)
        df["delta_temp"] = df["delta_temp"].fillna(0.0)

        # ---------------------------------------------------------
        # Collapse Outcome (NEW definition)
        # ---------------------------------------------------------
        df = df.sort_values(["problem_id", "attempt"])

        df["collapse_next"] = (
            df.groupby("problem_id")["collapse_flag"]
            .shift(-1)
            .fillna(0)
            .astype(int)
        )

        # Optional horizon outcome
        df["collapse_within_3"] = (
            df.groupby("problem_id")["collapse_flag"]
            .rolling(3)
            .max()
            .reset_index(level=0, drop=True)
            .shift(-1)
            .fillna(0)
            .astype(int)
        )

        return df

    # ---------------------------------------------------------
    # Estimators
    # ---------------------------------------------------------

    def _naive_difference(self, df):

        treated = df[df["treatment"] == 1]["collapsed_next"]
        control = df[df["treatment"] == 0]["collapsed_next"]

        ate = treated.mean() - control.mean()

        return {
            "ate": float(ate),
            "ci_lower": None,
            "ci_upper": None,
        }

    def _ipw(self, df):

        T = df["treatment"].values
        Y = df["collapsed_next"].values
        e = df["propensity"].values

        ipw = T * Y / e - (1 - T) * Y / (1 - e)

        ate = np.mean(ipw)
        se = np.std(ipw) / np.sqrt(len(ipw))

        return {
            "ate": float(ate),
            "ci_lower": float(ate - 1.96 * se),
            "ci_upper": float(ate + 1.96 * se),
        }

    def _cross_fitted_dr(self, df):

        feature_cols = [
            "total_energy",
            "energy_slope",
            "consecutive_hard",
            "consecutive_rising",
            "iteration",
            "temperature",
        ]

        X = df[feature_cols].values
        T = df["treatment_any"].values
        Y = df["collapse_within_3"].values

        kf = KFold(n_splits=5, shuffle=True, random_state=42)

        dr_scores = []

        for train_idx, test_idx in kf.split(X):

            # Outcome model
            mu_model = GradientBoostingRegressor()
            mu_model.fit(X[train_idx], Y[train_idx])
            mu_hat = mu_model.predict(X[test_idx])

            # Treatment model (estimated propensity)
            e_model = LogisticRegression(max_iter=500)
            e_model.fit(X[train_idx], T[train_idx])
            e_hat = e_model.predict_proba(X[test_idx])[:, 1]

            e_hat = np.clip(e_hat, 1e-3, 1 - 1e-3)

            correction = (
                T[test_idx] * (Y[test_idx] - mu_hat) / e_hat
            )

            dr = mu_hat + correction
            dr_scores.extend(dr)

        ate = np.mean(dr_scores)
        se = np.std(dr_scores) / np.sqrt(len(dr_scores))

        return {
            "ate": float(ate),
            "ci_lower": float(ate - 1.96 * se),
            "ci_upper": float(ate + 1.96 * se),
        }


def main():

    memory = Memory()
    evaluator = CausalEvaluationService(memory)

    # ---------------------------------------------------------
    # Get last 5 runs that actually have steps
    # ---------------------------------------------------------
    runs = memory.runs.get_recent_runs(limit=6)

    print(f"Found {len(runs)} recent runs")

    for run in runs:
        run_id = run.run_id

        # Skip incomplete runs
        if run.status != "completed":
            print(f"Skipping {run_id} (status={run.status})")
            continue

        # Ensure steps exist
        steps = memory.steps.get_by_run(run_id)
        if not steps:
            print(f"Skipping {run_id} (no steps)")
            continue

        print(f"\n🔍 Evaluating run: {run_id}")

        try:
            results = evaluator.evaluate_run(run_id)

            if results is None:
                print("No causal dataset built.")
                continue

            print("Results:")
            for method, r in results.items():
                print(f"  {method}: ATE={r['ate']:.4f}  CI=({r['ci_lower']}, {r['ci_upper']})")

        except Exception as e:
            print(f"❌ Failed evaluating {run_id}: {e}")


if __name__ == "__main__":
    main()