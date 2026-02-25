# constrain/evaluation/causal/evaluator.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import pandas as pd

from . import columns as C
from .dataset_builder import CausalDatasetBuilder  # existing :contentReference[oaicite:2]{index=2}
from .dataset_from_memory import build_causal_df_from_memory
from .estimators import CausalEstimators


@dataclass
class CausalEvalConfig:
    treatment_col: str = C.TREATMENT_ANY
    outcome_col: str = C.OUTCOME_H3

    feature_cols: tuple[str, ...] = (
        C.ENERGY,
        C.SLOPE,
        C.HARD,
        C.RISING,
        C.ITER,
        C.TEMP,
    )


class CausalEvaluator:
    """
    Canonical causal evaluation.

    - Builds dataframe (from memory or provided)
    - Builds (X,T,Y)
    - Runs naive/ipw/dr
    - Returns dict ready to persist
    """

    def __init__(self, *, n_splits: int = 5, seed: int = 42):
        self.builder = CausalDatasetBuilder()
        self.estimators = CausalEstimators(n_splits=n_splits, seed=seed)

    def evaluate_df(
        self,
        df: pd.DataFrame,
        cfg: Optional[CausalEvalConfig] = None,
    ) -> Optional[Dict[str, Any]]:
        cfg = cfg or CausalEvalConfig()

        # require columns
        required = list(cfg.feature_cols) + [cfg.treatment_col, cfg.outcome_col]
        missing = [c for c in required if c not in df.columns]
        if missing:
            return {
                "skipped": True,
                "reason": f"missing columns: {missing}",
                "n_rows": int(len(df)),
            }

        X, T, Y = self.builder.build(
            df=df,
            feature_cols=list(cfg.feature_cols),
            treatment_col=cfg.treatment_col,
            outcome_col=cfg.outcome_col,
        )

        if len(Y) < 20:
            return {
                "skipped": True,
                "reason": "too few samples after dropna",
                "n_rows": int(len(df)),
                "n_used": int(len(Y)),
            }

        res = self.estimators.estimate_all(X=X, T=T, Y=Y)
        out = self.estimators.to_dict(res)
        out["skipped"] = False
        out["n_rows"] = int(len(df))
        out["n_used"] = int(len(Y))
        out["treatment_col"] = cfg.treatment_col
        out["outcome_col"] = cfg.outcome_col
        out["feature_cols"] = list(cfg.feature_cols)
        return out

    def evaluate_run(
        self,
        *,
        memory,
        run_id: str,
        cfg: Optional[CausalEvalConfig] = None,
    ) -> Optional[Dict[str, Any]]:
        df = build_causal_df_from_memory(memory, run_id)
        if df.empty:
            return None
        return self.evaluate_df(df, cfg=cfg)