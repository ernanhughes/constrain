# analysis/causal/dataset_builder.py

from __future__ import annotations
import pandas as pd
import numpy as np


class CausalDatasetBuilder:

    def build(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
        treatment_col: str,
        outcome_col: str,
    ):

        df = df.dropna(
            subset=feature_cols + [treatment_col, outcome_col]
        )

        X = df[feature_cols].values
        T = df[treatment_col].values.astype(int)
        Y = df[outcome_col].values.astype(float)

        return X, T, Y