# constrain/analysis/signal_discovery_service.py

from __future__ import annotations

import json
import time
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import TimeSeriesSplit

from constrain.analysis.aggregation.metrics_aggregator import MetricsAggregator
from constrain.data.memory import Memory
from constrain.data.schemas.signal_report import SignalReportDTO

import logging
logger = logging.getLogger(__name__)

class SignalDiscoveryService:

    def __init__(self, memory: Memory):
        self.memory = memory

    # ---------------------------------------------------------
    # Public API
    # ---------------------------------------------------------

    def analyze_run(self, run_id: str, prediction_horizon: int = 2, min_samples: int = 50):
        full_df = MetricsAggregator.build_run_dataframe(self.memory, run_id)
        
        # ✅ Early exit if insufficient data
        if len(full_df) < min_samples:
            logger.warning(f"⚠️ Skipping signal discovery: only {len(full_df)} steps (need {min_samples})")
            return {
                "skipped": True,
                "reason": f"insufficient_samples ({len(full_df)} < {min_samples})",
                "dataframe": full_df,
            }

        df = self._prepare_labels(full_df.copy(), prediction_horizon)
        feature_cols = self._select_feature_columns(df)
        target_col = f"collapse_t_plus_{prediction_horizon}"

        df_clean = df.dropna(subset=feature_cols + [target_col])

        # ✅ Check after cleaning
        if len(df_clean) < min_samples:
            logger.warning(f"⚠️ Skipping signal discovery: only {len(df_clean)} valid rows after cleaning")
            return {
                "skipped": True,
                "reason": f"insufficient_valid_rows ({len(df_clean)} < {min_samples})",
                "dataframe": full_df,
            }

        # ✅ Reduce folds for small datasets
        n_splits = min(5, len(df_clean) // 10)  # At least 10 samples per fold
        if n_splits < 2:
            logger.warning(f"⚠️ Skipping signal discovery: not enough data for {n_splits} folds")
            return {"skipped": True, "reason": "insufficient_folds", "dataframe": full_df}

        model, auc_score, fold_aucs = self._train_model(
            df_clean, feature_cols, target_col=target_col, n_splits=n_splits
        )
        importance = self._extract_importance(model, feature_cols)

        diagnostics = self._compute_diagnostics(df_clean, target_col)

        return {
            "dataframe": full_df,
            "model_results": {
                "auc_mean": float(auc_score),
                "auc_per_fold": fold_aucs,
                "feature_importance": importance,
                "features_used": feature_cols,
                "prediction_horizon": prediction_horizon,
                "num_samples": len(df_clean),
            },
            "diagnostics": diagnostics,
        }

    # ---------------------------------------------------------
    # Label Engineering
    # ---------------------------------------------------------

    def _prepare_labels(self, df: pd.DataFrame, horizon: int) -> pd.DataFrame:
        df = df.sort_values(["problem_id", "iteration"])

        df["collapse_label"] = (df["phase"] == "collapse").astype(int)

        df[f"collapse_t_plus_{horizon}"] = (
            df.groupby(["problem_id"])["collapse_label"]
            .shift(-horizon)
        )

        return df

    # ---------------------------------------------------------
    # Feature Selection
    # ---------------------------------------------------------

    def _select_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """
        Select numeric columns suitable for modeling.
        """

        excluded = {
            "step_id",
            "run_id",
            "problem_id",
            "phase",
            "collapse_label",
        }

        numeric_cols = df.select_dtypes(include=[np.number]).columns

        return [c for c in numeric_cols if c not in excluded]

    # ---------------------------------------------------------
    # Model Training
    # ---------------------------------------------------------

    def _train_model(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        target_col: str,
        n_splits: int = 5,
    ):

        tscv = TimeSeriesSplit(n_splits=n_splits)

        aucs = []
        models = []

        X = df[feature_cols]
        y = df[target_col]

        for train_idx, test_idx in tscv.split(df):

            X_train = X.iloc[train_idx]
            y_train = y.iloc[train_idx]

            X_test = X.iloc[test_idx]
            y_test = y.iloc[test_idx]

            model = xgb.XGBClassifier(
                n_estimators=150,
                max_depth=4,
                learning_rate=0.08,
                subsample=0.8,
                colsample_bytree=0.8,
                eval_metric="logloss",
                random_state=42,
            )

            model.fit(X_train, y_train)

            if len(set(y_test)) < 2:
                auc = float("nan")
            else:
                y_pred = model.predict_proba(X_test)[:, 1]
                auc = roc_auc_score(y_test, y_pred)

            aucs.append(float(auc))
            models.append(model)

        avg_auc = float(np.nanmean(aucs))

        return models[-1], avg_auc, aucs

    # ---------------------------------------------------------
    # Importance Extraction
    # ---------------------------------------------------------

    def _extract_importance(
        self,
        model,
        feature_cols: List[str],
    ) -> List[Dict[str, Any]]:

        importances = model.feature_importances_

        ranking = sorted(
            zip(feature_cols, importances),
            key=lambda x: x[1],
            reverse=True,
        )

        return [
            {"feature": f, "importance": float(score)}
            for f, score in ranking
        ]

    def analyze_and_persist(self, run_id: str):

        results = self.analyze_run(run_id)

        dto = SignalReportDTO(
            run_id=run_id,
            mean_energy=results["diagnostics"]["mean_energy"],
            mean_energy_slope=results["diagnostics"]["mean_energy_slope"],
            intervention_recovery_delta=results["diagnostics"]["intervention_recovery_delta"],
            auc=results["model_results"]["auc_mean"],
            feature_importance_json=json.dumps(
                results["model_results"]["feature_importance"]
            ),
            created_at=time.time(),
        )

        self.memory.signal_reports.create(dto)

        return results
    
    def _compute_diagnostics(self, df: pd.DataFrame, target_col: str):

        diagnostics = {}

        diagnostics["mean_energy"] = float(df["total_energy"].mean())

        if "total_energy" in df.columns:
            slope = (
                df.groupby("problem_id")["total_energy"]
                .diff()
                .mean()
            )
            diagnostics["mean_energy_slope"] = float(slope)
        else:
            diagnostics["mean_energy_slope"] = None

        if "policy_action" in df.columns:
            intervention_rows = df[df["policy_action"] != "ACCEPT"]
            diagnostics["intervention_recovery_delta"] = float(
                intervention_rows["total_energy"].mean()
            ) if not intervention_rows.empty else None
        else:
            diagnostics["intervention_recovery_delta"] = None

        label_counts = df[target_col].value_counts().to_dict()
        diagnostics["label_distribution"] = {
            str(k): int(v) for k, v in label_counts.items()
        }

        return diagnostics
