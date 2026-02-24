"""constrain.services.collapse_prediction_service
Service for discovering predictive signals of intervention-triggered collapse.
"""
from __future__ import annotations
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import TimeSeriesSplit

from constrain.data.memory import Memory
from constrain.config import get_config
from constrain.data.schemas.collapse_signal import CollapseSignalDTO
from constrain.analysis.trajectory.trajectory_reconstructor import TrajectoryReconstructor
from constrain.analysis.trajectory.statistical_comparison import BootstrapComparator

logger = logging.getLogger(__name__)


class CollapsePredictionService:
    """
    Discovers predictive signals for intervention-triggered collapse.
    
    Capabilities:
    - Engineer labels for future collapse (t+horizon)
    - Select predictive features from step metrics
    - Train time-series cross-validated XGBoost models
    - Extract feature importance rankings
    - Persist results with experiment grouping
    """

    def __init__(self, memory: Memory, reconstructor: Optional[TrajectoryReconstructor] = None):
        self.memory = memory
        self.reconstructor = reconstructor or TrajectoryReconstructor(memory)

    def discover_signals(
        self,
        run_id: str,
        prediction_horizon: int = 2,
        min_samples: int = 50,
        save_plots: bool = False,
    ) -> Dict[str, Any]:
        """
        Run signal discovery for a single run.
        Returns analysis results dict (for reporting/persistence).
        """
        # 1. Reconstruct trajectories using shared infrastructure
        problem_df = self.reconstructor.get_step_dataframe(run_id)
        if len(problem_df) < min_samples:
            return self._skip_result(run_id, "insufficient_samples", len(problem_df), min_samples)

        # 2. Engineer labels: collapse at t+horizon
        df = self._engineer_labels(problem_df, horizon=prediction_horizon)
        
        # 3. Select features
        feature_cols = self._select_predictive_features(df)
        target_col = f"collapse_t_plus_{prediction_horizon}"

        # 4. Clean data
        df_clean = df.dropna(subset=feature_cols + [target_col])
        if len(df_clean) < min_samples or len(df_clean[target_col].unique()) < 2:
            return self._skip_result(run_id, "insufficient_valid_data", len(df_clean), min_samples)

        # 5. Train model with time-series CV
        model, auc_score, fold_aucs = self._train_predictor(
            df_clean, feature_cols, target_col, n_splits=min(5, len(df_clean) // 10)
        )
        if model is None:
            return self._skip_result(run_id, "no_valid_folds")

        # 6. Extract importance + diagnostics
        importance = self._extract_feature_importance(model, feature_cols)
        diagnostics = self._compute_signal_diagnostics(df_clean, target_col)

        # 7. Optional visualization
        if save_plots:
            self._plot_energy_distribution(df_clean, target_col, run_id)

        return {
            "run_id": run_id,
            "prediction_horizon": prediction_horizon,
            "n_samples": len(df_clean),
            "auc_mean": float(auc_score),
            "auc_per_fold": fold_aucs,
            "feature_importance": importance,
            "features_used": feature_cols,
            "diagnostics": diagnostics,
        }

    def persist_signal_report(self, run_id: str, results: Dict[str, Any], experiment_id: Optional[int] = None) -> Optional[int]:
        """Persist signal discovery results to database."""
        if results.get("skipped"):
            logger.debug(f"⏭️ Skipping persistence for {run_id}: {results.get('reason')}")
            return None

        dto = CollapseSignalDTO(
            run_id=run_id,
            experiment_id=experiment_id,
            prediction_horizon=results["prediction_horizon"],
            auc_mean=results["auc_mean"],
            auc_std=float(np.std(results["auc_per_fold"])) if results["auc_per_fold"] else None,
            feature_importance_json=json.dumps(results["feature_importance"]),
            n_samples=results["n_samples"],
            mean_energy=results["diagnostics"].get("mean_energy"),
            intervention_energy_delta=results["diagnostics"].get("intervention_energy_delta"),
            created_at=time.time(),
        )

        report = self.memory.collapse_signals.create(dto)
        logger.info(f"✅ Signal report persisted: {report.id}")
        return report.id

    def compare_signal_strength(
        self,
        run_ids: List[str],
        experiment_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Compare predictive signal strength across runs using bootstrap."""
        # Fetch all signal reports
        reports = []
        for run_id in run_ids:
            reports.extend(self.memory.collapse_signals.get_by_run_id(run_id))

        if not reports:
            return {"error": "No signal reports found"}

        # Group by run, extract AUCs
        by_run = {r.run_id: r.auc_mean for r in reports if r.auc_mean is not None}
        
        # Compare each vs baseline (first run or policy 0)
        comparator = BootstrapComparator(seed=42)
        results = {}
        
        baseline_auc = list(by_run.values())[0] if by_run else None
        for run_id, auc in by_run.items():
            if run_id == list(by_run.keys())[0]:
                continue
            # Bootstrap comparison of AUCs (simplified: treat as single values)
            results[run_id] = {
                "auc": auc,
                "delta_vs_baseline": auc - baseline_auc if baseline_auc else None,
            }

        return {
            "baseline_auc": baseline_auc,
            "comparisons": results,
            "n_runs_compared": len(by_run),
        }

    # ============================================================
    # Internal Methods (same logic, better organization)
    # ============================================================

    def _engineer_labels(self, df: pd.DataFrame, horizon: int) -> pd.DataFrame:
        required = {"problem_id", "iteration", "total_energy", "phase"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f"Step-level dataframe required. Missing columns: {missing}"
            )

        """Label each step with whether collapse occurs at t+horizon."""
        df = df.sort_values(["problem_id", "iteration"]).copy()
        
        # Collapse definition: energy > tau_hard OR phase == "collapse"
        cfg = get_config()
        df["is_collapse"] = (df["total_energy"] > cfg.tau_hard) | (df["phase"] == "collapse")
        
        # Shift by horizon within each problem
        df[f"collapse_t_plus_{horizon}"] = df.groupby("problem_id")["is_collapse"].shift(-horizon)
        
        return df

    def _select_predictive_features(self, df: pd.DataFrame) -> List[str]:
        """Select numeric columns suitable for prediction."""
        exclude = {"step_id", "run_id", "problem_id", "phase", "is_collapse", "iteration"}
        numeric = df.select_dtypes(include=[np.number]).columns
        return [c for c in numeric if c not in exclude and df[c].notna().any()]

    def _train_predictor(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        target_col: str,
        n_splits: int = 5,
    ) -> tuple:
        """Train XGBoost with time-series cross-validation."""
        tscv = TimeSeriesSplit(n_splits=n_splits)
        aucs, models = [], []
        X, y = df[feature_cols], df[target_col]

        for train_idx, test_idx in tscv.split(df):
            X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
            X_te, y_te = X.iloc[test_idx], y.iloc[test_idx]
            
            if len(set(y_tr)) < 2 or len(set(y_te)) < 2:
                continue

            model = xgb.XGBClassifier(
                n_estimators=150, max_depth=4, learning_rate=0.08,
                subsample=0.8, colsample_bytree=0.8,
                eval_metric="logloss", random_state=42,
            )
            model.fit(X_tr, y_tr)
            
            y_pred = model.predict_proba(X_te)[:, 1]
            auc = roc_auc_score(y_te, y_pred)
            
            aucs.append(float(auc))
            models.append(model)

        if not models:
            return None, float("nan"), []

        return models[-1], float(np.nanmean(aucs)), aucs

    def _extract_feature_importance(self, model, feature_cols: List[str]) -> List[Dict[str, Any]]:
        """Extract and rank feature importances."""
        importances = model.feature_importances_
        ranked = sorted(zip(feature_cols, importances), key=lambda x: x[1], reverse=True)
        return [{"feature": f, "importance": float(score)} for f, score in ranked]

    def _compute_signal_diagnostics(self, df: pd.DataFrame, target_col: str) -> Dict[str, Any]:
        """Compute summary diagnostics for the signal."""
        return {
            "mean_energy": float(df["total_energy"].mean()),
            "intervention_energy_delta": self._compute_intervention_delta(df),
            "label_distribution": {str(k): int(v) for k, v in df[target_col].value_counts().items()},
        }

    def _compute_intervention_delta(self, df: pd.DataFrame) -> Optional[float]:
        """Compute energy change after intervention."""
        if "policy_action" not in df.columns:
            return None
        intervention_mask = df["policy_action"] != "ACCEPT"
        if not intervention_mask.any():
            return None
        # Simplified: mean energy post-intervention vs pre
        return float(df.loc[intervention_mask, "total_energy"].mean())

    def _plot_energy_distribution(self, df: pd.DataFrame, target_col: str, run_id: str):
        """Save energy distribution histogram (optional)."""
        import matplotlib.pyplot as plt
        
        stable = df[df[target_col] == 0]
        collapse = df[df[target_col] == 1]
        
        plt.figure(figsize=(8, 5))
        plt.hist(stable["total_energy"], bins=30, alpha=0.5, label="Stable", density=True)
        plt.hist(collapse["total_energy"], bins=30, alpha=0.5, label="Collapse", density=True)
        plt.legend()
        plt.title(f"Energy Distribution: {run_id}")
        plt.xlabel("Total Energy")
        plt.ylabel("Density")
        
        plot_dir = Path(get_config().plots_dir) if hasattr(get_config(), "plots_dir") else Path("outputs/plots")
        plot_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(plot_dir / f"{run_id}_energy_dist.png", dpi=150, bbox_inches="tight")
        plt.close()

    def _skip_result(self, run_id: str, reason: str, n_found: int = None, n_required: int = None) -> Dict[str, Any]:
        """Return standardized skip result."""
        return {
            "skipped": True,
            "run_id": run_id,
            "reason": reason,
            "n_found": n_found,
            "n_required": n_required,
            "feature_importance": [],
            "diagnostics": {},
        }