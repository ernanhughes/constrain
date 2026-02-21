# src/constrain/analysis/stage3/engine/feature_engineering.py

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd


class FeatureEngineer:

    def __init__(
        self,
        variance_threshold: float = 1e-8,
        remove_constant: bool = True,
    ):
        self.variance_threshold = variance_threshold
        self.remove_constant = remove_constant

    def prepare(
        self,
        df: pd.DataFrame,
        target_col: str,
        exclude_cols: List[str],
    ) -> Tuple[pd.DataFrame, pd.Series]:

        if target_col not in df.columns:
            raise ValueError(f"{target_col} not found in dataframe.")

        df = df.dropna(subset=[target_col])

        if df[target_col].nunique() < 2:
            raise ValueError("Target has only one class.")

        numeric_cols = df.select_dtypes(include=[np.number]).columns

        exclude = set(exclude_cols) | {target_col}

        feature_cols = [
            c for c in numeric_cols
            if c not in exclude
        ]

        X = df[feature_cols].copy()
        y = df[target_col]

        if self.remove_constant:
            nunique = X.nunique()
            constant_cols = nunique[nunique <= 1].index
            X = X.drop(columns=constant_cols)

        low_var_cols = X.var()[X.var() < self.variance_threshold].index
        X = X.drop(columns=low_var_cols)

        return X, y