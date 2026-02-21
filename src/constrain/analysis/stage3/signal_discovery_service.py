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
from constrain.utils.json_utils import dumps_safe

import logging

logger = logging.getLogger(__name__)


class SignalDiscoveryService:
    def __init__(self, memory: Memory):
        self.memory = memory

    # ---------------------------------------------------------
    # Public API
    # ---------------------------------------------------------

    def analyze_run(
        self, run_id: str, prediction_horizon: int = 2, min_samples: int = 50
    ):
        full_df = MetricsAggregator.build_run_dataframe(self.memory, run_id)

        # ✅ Early exit if insufficient data
        if len(full_df) < min_samples:
            logger.warning(
                f"⚠️ Skipping signal discovery: only {len(full_df)} steps (need {min_samples})"
            )
            return {
                "skipped": True,
                "reason": f"insufficient_samples ({len(full_df)} < {min_samples})",
                "dataframe": full_df,
                "diagnostics": {},
                "model_results": {},
            }

        df = self._prepare_labels(full_df.copy(), prediction_horizon)
        feature_cols = self._select_feature_columns(df)
        target_col = f"collapse_t_plus_{prediction_horizon}"

        df_clean = df.dropna(subset=feature_cols + [target_col])
        label_counts = df_clean[target_col].value_counts()

        if len(label_counts) < 2:
            logger.warning(
                f"⚠️ Skipping signal discovery: only one class present "
                f"in target ({label_counts.to_dict()})"
            )
            return {
                "skipped": True,
                "reason": "single_class_target",
                "dataframe": full_df,
                "diagnostics": {
                    "label_distribution": {
                        str(k): int(v) for k, v in label_counts.items()
                    }
                },
                "model_results": {},
            }


        # ✅ Check after cleaning
        if len(df_clean) < min_samples:
            logger.warning(
                f"⚠️ Skipping signal discovery: only {len(df_clean)} valid rows after cleaning"
            )
            return {
                "skipped": True,
                "reason": f"insufficient_valid_rows ({len(df_clean)} < {min_samples})",
                "dataframe": full_df,
                "diagnostics": {},
                "model_results": {},
            }

        # ✅ Reduce folds for small datasets
        n_splits = min(5, len(df_clean) // 10)  # At least 10 samples per fold
        if n_splits < 2:
            logger.warning(
                f"⚠️ Skipping signal discovery: not enough data for {n_splits} folds"
            )
            return {
                "skipped": True,
                "reason": "insufficient_folds",
                "dataframe": full_df,
                "diagnostics": {},
                "model_results": {},
           }

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


    def analyze_and_persist(self, run_id: str):
        """
        Run signal discovery analysis and persist results to database.
        
        Returns:
            dict: Analysis results (same structure as analyze_run)
        """
        # 1. Run the analysis
        results = self.analyze_run(run_id)
        
        # 2. ✅ Skip persistence if run was skipped due to insufficient data
        if results.get("skipped"):
            logger.debug(f"⏭️ Skipping persistence for {run_id}: {results.get('reason')}")
            return results
        
        # 3. ✅ Safe access with defaults for robustness
        diagnostics = results.get("diagnostics", {})
        model_results = results.get("model_results", {})
        
        # 4. ✅ Create DTO with safe .get() access to avoid KeyError
        dto = SignalReportDTO(
            run_id=run_id,
            mean_energy=diagnostics.get("mean_energy"),
            mean_energy_slope=diagnostics.get("mean_energy_slope"),
            intervention_recovery_delta=diagnostics.get("intervention_recovery_delta"),
            auc=model_results.get("auc_mean"),
            feature_importance_json=json.dumps(
                model_results.get("feature_importance", [])
            ),
            created_at=time.time(),
        )
        
        # 5. ✅ Persist to database
        try:
            self.memory.signal_reports.create(dto)
            logger.debug(f"✅ Signal report persisted for {run_id}")
        except Exception as e:
            logger.exception(f"⚠️ Failed to persist signal report for {run_id}: {e}")
            # Don't crash the pipeline — return results anyway
            return results
        
        return results

    # ---------------------------------------------------------
    # Label Engineering
    # ---------------------------------------------------------

    def _prepare_labels(self, df: pd.DataFrame, horizon: int) -> pd.DataFrame:
        df = df.sort_values(["problem_id", "iteration"])

        df["collapse_label"] = (df["phase"] == "collapse").astype(int)

        df[f"collapse_t_plus_{horizon}"] = df.groupby(["problem_id"])[
            "collapse_label"
        ].shift(-horizon)

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

            # 🔥 Skip invalid folds
            if len(set(y_train)) < 2:
                continue

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

        if not models:
            logger.warning("⚠️ All folds skipped — no valid training splits")
            return None, float("nan"), []

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

        return [{"feature": f, "importance": float(score)} for f, score in ranking]

    def _compute_diagnostics(self, df: pd.DataFrame, target_col: str):
        diagnostics = {}

        diagnostics["mean_energy"] = float(df["total_energy"].mean())

        if "total_energy" in df.columns:
            slope = df.groupby("problem_id")["total_energy"].diff().mean()
            diagnostics["mean_energy_slope"] = float(slope)
        else:
            diagnostics["mean_energy_slope"] = None

        if "policy_action" in df.columns:
            intervention_rows = df[df["policy_action"] != "ACCEPT"]
            diagnostics["intervention_recovery_delta"] = (
                float(intervention_rows["total_energy"].mean())
                if not intervention_rows.empty
                else None
            )
        else:
            diagnostics["intervention_recovery_delta"] = None

        label_counts = df[target_col].value_counts().to_dict()
        diagnostics["label_distribution"] = {
            str(k): int(v) for k, v in label_counts.items()
        }

        return diagnostics
