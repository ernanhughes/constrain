from __future__ import annotations

from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

from .calibration import ProbabilityCalibration
from .stability import SignalStabilityAnalyzer
from .temperature_scaling import TemperatureScaler
from .bootstrap import BootstrapCI
from constrain.analysis.stage3.registry.model_registry import ModelRegistry
from constrain.analysis.stage3.registry.leaderboard import Leaderboard
from constrain.analysis.stage3.engine.shap_explainer import ShapExplainer


class SignalModelingEngine:

    def __init__(
        self,
        n_splits: int = 5,
        random_state: int = 42,
    ):
        self.n_splits = n_splits
        self.random_state = random_state

    # =========================================================
    # PUBLIC ENTRY
    # =========================================================

    def run(
        self,
        df: pd.DataFrame,
        target_col: str,
        group_col: str = "problem_id",
        exclude_cols: List[str] | None = None,
        run_id: str | None = None,
        use_shap: bool = True,
        use_bootstrap: bool = True,
        use_temperature_scaling: bool = True,
        persist_model: bool = True,
        update_leaderboard: bool = True,
    ) -> Dict[str, Any]:

        if exclude_cols is None:
            exclude_cols = []

        # -----------------------------
        # Feature Preparation
        # -----------------------------

        X, y, groups = self._prepare_features(
            df,
            target_col,
            group_col,
            exclude_cols,
        )

        if y.nunique() < 2:
            raise ValueError("Target has only one class.")

        # -----------------------------
        # Cross Validation
        # -----------------------------

        gkf = GroupKFold(n_splits=self.n_splits)

        aucs = []
        feature_rankings = []
        prob_outputs: List[Tuple[np.ndarray, np.ndarray]] = []

        oof_y = []
        oof_probs = []

        for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups)):

            model = self._model_factory()

            X_train = X.iloc[train_idx]
            y_train = y.iloc[train_idx]

            X_test = X.iloc[test_idx]
            y_test = y.iloc[test_idx]

            if y_test.nunique() < 2:
                continue

            model.fit(X_train, y_train)

            logits = model.predict(X_test, output_margin=True)

            # ---------------------------------
            # Optional Temperature Scaling
            # ---------------------------------

            if use_temperature_scaling:
                scaler = TemperatureScaler().fit(logits, y_test.values)
                probs = scaler.transform(logits)
            else:
                probs = model.predict_proba(X_test)[:, 1]

            auc = roc_auc_score(y_test, probs)
            aucs.append(float(auc))

            feature_rankings.append(
                sorted(
                    zip(X.columns, model.feature_importances_),
                    key=lambda x: x[1],
                    reverse=True,
                )
            )

            prob_outputs.append((y_test.values, probs))

            oof_y.extend(y_test.values)
            oof_probs.extend(probs)

        if not aucs:
            raise ValueError("No valid folds produced AUC.")

        # -----------------------------
        # Stability Diagnostics
        # -----------------------------

        auc_stats = SignalStabilityAnalyzer.auc_stability(aucs)

        feature_stability = SignalStabilityAnalyzer.feature_stability(
            feature_rankings
        )

        calibration_stats = [
            ProbabilityCalibration.analyze(y_true, y_probs)
            for y_true, y_probs in prob_outputs
        ]

        # -----------------------------
        # Bootstrap CI
        # -----------------------------

        bootstrap_stats = None
        if use_bootstrap:
            bootstrap_stats = BootstrapCI.auc_ci(oof_y, oof_probs)

        # -----------------------------
        # Final Model Training
        # -----------------------------

        final_model = self._model_factory()
        final_model.fit(X, y)

        global_scaler = None
        if use_temperature_scaling:
            full_logits = final_model.predict(X, output_margin=True)
            global_scaler = TemperatureScaler().fit(full_logits, y.values)

        # -----------------------------
        # SHAP (optional)
        # -----------------------------

        shap_importance = None
        if use_shap:
            shap_importance = ShapExplainer.explain(final_model, X)

        # -----------------------------
        # Persistence
        # -----------------------------

        registry_path = None
        if persist_model and run_id is not None:
            registry_path = ModelRegistry.save(
                run_id=run_id,
                model=final_model,
                scaler=global_scaler,
                metadata={
                    "mean_auc": auc_stats["mean_auc"],
                    "std_auc": auc_stats["std_auc"],
                    "ci": bootstrap_stats,
                    "num_features": X.shape[1],
                    "n_splits": self.n_splits,
                },
            )

        # -----------------------------
        # Leaderboard
        # -----------------------------

        if update_leaderboard and run_id is not None:
            Leaderboard.update(
                run_id,
                {
                    "mean_auc": auc_stats["mean_auc"],
                    "ci_lower": bootstrap_stats["ci_lower"] if bootstrap_stats else None,
                    "ci_upper": bootstrap_stats["ci_upper"] if bootstrap_stats else None,
                    "num_features": int(X.shape[1]),
                },
            )

        # -----------------------------
        # Return Results
        # -----------------------------

        result = {
            "mean_auc": auc_stats["mean_auc"],
            "std_auc": auc_stats["std_auc"],
            "cv_ratio": auc_stats["cv_ratio"],
            "bootstrap_ci": bootstrap_stats,
            "feature_importance": feature_rankings[-1],
            "feature_stability": feature_stability,
            "calibration_per_fold": calibration_stats,
            "shap_importance": shap_importance,
            "num_features": int(X.shape[1]),
            "num_folds": int(len(aucs)),
            "registry_path": registry_path,
        }

        return self._make_json_safe(result)

    # =========================================================
    # Helpers
    # =========================================================

    def _model_factory(self):
        return XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method="hist",
            eval_metric="logloss",
            random_state=self.random_state,
        )

    def _prepare_features(
        self,
        df: pd.DataFrame,
        target_col: str,
        group_col: str,
        exclude_cols: List[str],
    ):

        if target_col not in df.columns:
            raise ValueError(f"{target_col} not found.")

        if group_col not in df.columns:
            raise ValueError(f"{group_col} not found.")

        df = df.dropna(subset=[target_col])

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        exclude = set(exclude_cols) | {target_col}

        feature_cols = [c for c in numeric_cols if c not in exclude]

        X = df[feature_cols].copy()

        # Remove constant columns
        nunique = X.nunique()
        X = X.drop(columns=nunique[nunique <= 1].index)

        # Remove near-zero variance
        X = X.drop(columns=X.var()[X.var() < 1e-8].index)

        y = df[target_col]
        groups = df[group_col]

        return X, y, groups

    def _make_json_safe(self, obj):

        import numpy as np

        if isinstance(obj, dict):
            return {k: self._make_json_safe(v) for k, v in obj.items()}

        if isinstance(obj, list):
            return [self._make_json_safe(v) for v in obj]

        if isinstance(obj, tuple):
            return [self._make_json_safe(v) for v in obj]

        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)

        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)

        return obj