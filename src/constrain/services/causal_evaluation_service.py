"""constrain.services.causal_evaluation_service
Service for evaluating causal effects of interventions in runs.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold


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

    # ---------------------------------------------------------
    # Build Causal Dataset
    # ---------------------------------------------------------

    def _build_causal_dataframe(self, run_id):

        steps = self.memory.steps.get_by_run(run_id)
        events = self.memory.policy_events.get_by_run_id(run_id)

        if not steps or not events:
            return pd.DataFrame()

        steps_df = pd.DataFrame([s.model_dump() for s in steps])
        events_df = pd.DataFrame([e.model_dump() for e in events])

        df = steps_df.merge(
            events_df,
            on="step_id",
            how="inner",
        )

        # Treatment: any intervention
        df["treatment"] = (df["action"] != "ACCEPT").astype(int)

        # Outcome: collapse next step
        df = df.sort_values(["problem_id", "iteration"])

        df["collapsed_next"] = (
            df.groupby("problem_id")["total_energy"]
            .shift(-1)
            > df["tau_hard"]
        ).astype(int)

        df = df.dropna(subset=["collapsed_next"])

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
            "stability_energy",
            "iteration",
            "temperature",
        ]

        X = df[feature_cols].values
        T = df["treatment"].values
        Y = df["collapsed_next"].values
        e_logged = df["propensity"].values

        kf = KFold(n_splits=5, shuffle=True, random_state=42)

        dr_scores = []

        for train_idx, test_idx in kf.split(X):

            mu_model = GradientBoostingRegressor()
            mu_model.fit(X[train_idx], Y[train_idx])

            mu_hat = mu_model.predict(X[test_idx])

            correction = (
                T[test_idx] * (Y[test_idx] - mu_hat)
                / np.clip(e_logged[test_idx], 1e-3, 1)
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