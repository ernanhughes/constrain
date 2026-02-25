# constrain/evaluation/causal/naive_estimator.py
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class NaiveResult:
    ate: float
    n: int
    treated_rate: float


class NaiveATEEstimator:
    """
    Naive difference-in-means:
        ATE = E[Y|T=1] - E[Y|T=0]

    No propensity model. No outcome model.
    """
    def estimate(self, T: np.ndarray, Y: np.ndarray) -> NaiveResult:
        T = T.astype(int)
        Y = Y.astype(float)

        treated = Y[T == 1]
        control = Y[T == 0]

        if len(treated) == 0 or len(control) == 0:
            return NaiveResult(
                ate=float("nan"),
                n=int(len(Y)),
                treated_rate=float(T.mean()) if len(Y) else float("nan"),
            )

        ate = float(treated.mean() - control.mean())

        return NaiveResult(
            ate=ate,
            n=int(len(Y)),
            treated_rate=float(T.mean()),
        )