# analysis/causal/ipw_estimator.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np


@dataclass
class IPWResult:
    ate: float
    ate_stabilized: float
    std_error: float
    n: int
    overlap_min: float
    overlap_max: float
    treated_rate: float


class IPWEstimator:
    """
    Inverse Propensity Weighting estimator.

    Pure statistical component.
    No DB.
    No memory.
    No side effects.
    """

    def __init__(
        self,
        clip_min: float = 0.01,
        clip_max: float = 0.99,
        stabilized: bool = True,
    ):
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.stabilized = stabilized

    def estimate(
        self,
        T: np.ndarray,          # treatment (0/1)
        Y: np.ndarray,          # outcome
        propensity: np.ndarray  # P(T=1 | X)
    ) -> IPWResult:

        T = T.astype(float)
        Y = Y.astype(float)
        e = propensity.astype(float)

        # -------------------------------------------------
        # 1️⃣ Clip for overlap safety
        # -------------------------------------------------

        e = np.clip(e, self.clip_min, self.clip_max)

        # Diagnostics
        overlap_min = float(e.min())
        overlap_max = float(e.max())
        treated_rate = float(T.mean())

        # -------------------------------------------------
        # 2️⃣ Compute weights
        # -------------------------------------------------

        if self.stabilized:
            p_t = treated_rate
            w = (
                T * p_t / e
                + (1 - T) * (1 - p_t) / (1 - e)
            )
        else:
            w = (
                T / e
                + (1 - T) / (1 - e)
            )

        # -------------------------------------------------
        # 3️⃣ Estimate potential outcomes
        # -------------------------------------------------

        mu1 = np.sum(w * T * Y) / np.sum(w * T)
        mu0 = np.sum(w * (1 - T) * Y) / np.sum(w * (1 - T))

        ate = float(mu1 - mu0)

        # -------------------------------------------------
        # 4️⃣ Compute standard error (robust sandwich)
        # -------------------------------------------------

        influence = (
            w * (T - e) * (Y - (mu1 * T + mu0 * (1 - T)))
        )

        std_error = float(np.std(influence) / np.sqrt(len(Y)))

        return IPWResult(
            ate=ate,
            ate_stabilized=ate if self.stabilized else ate,
            std_error=std_error,
            n=len(Y),
            overlap_min=overlap_min,
            overlap_max=overlap_max,
            treated_rate=treated_rate,
        )