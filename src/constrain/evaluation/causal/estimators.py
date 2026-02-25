# constrain/evaluation/causal/estimators.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

from .dr_estimator import CrossFittedDREstimator, DRResult
from .ipw_estimator import IPWEstimator, IPWResult
from .naive_estimator import NaiveATEEstimator, NaiveResult
from .propensity_model import PropensityModel


@dataclass
class CausalDiagnostics:
    n: int
    treated_rate: float
    overlap_min: Optional[float] = None
    overlap_max: Optional[float] = None
    ess: Optional[float] = None


@dataclass
class CausalResults:
    naive: NaiveResult
    ipw: Optional[IPWResult]
    dr: DRResult
    diagnostics: CausalDiagnostics


class CausalEstimators:
    """
    Canonical estimation entry point.

    - Always returns Naive and DR.
    - IPW requires propensities; we estimate them unless passed in.
    """

    def __init__(
        self,
        *,
        n_splits: int = 5,
        seed: int = 42,
        clip_min: float = 0.01,
        clip_max: float = 0.99,
    ):
        self.naive = NaiveATEEstimator()
        self.dr = CrossFittedDREstimator(
            n_splits=n_splits,
            seed=seed,
            clip_min=clip_min,
            clip_max=clip_max,
        )
        self.ipw = IPWEstimator(
            clip_min=clip_min,
            clip_max=clip_max,
            stabilized=True,
        )
        self._prop_model = PropensityModel()

    def estimate_all(
        self,
        *,
        X: np.ndarray,
        T: np.ndarray,
        Y: np.ndarray,
        propensity: Optional[np.ndarray] = None,
    ) -> CausalResults:

        naive_res = self.naive.estimate(T, Y)

        # DR does its own cross-fitted propensities internally
        dr_res = self.dr.estimate(X, T, Y)

        # IPW: use provided propensity or estimate a single global propensity model
        ipw_res: Optional[IPWResult] = None
        if propensity is None:
            try:
                propensity = self._prop_model.fit_predict(X, T, X)
            except Exception:
                propensity = None

        if propensity is not None:
            ipw_res = self.ipw.estimate(T=T, Y=Y, propensity=propensity)

        diag = CausalDiagnostics(
            n=int(len(Y)),
            treated_rate=float(np.mean(T)) if len(T) else float("nan"),
            overlap_min=getattr(dr_res, "overlap_min", None),
            overlap_max=getattr(dr_res, "overlap_max", None),
            ess=getattr(dr_res, "ess", None),
        )

        return CausalResults(
            naive=naive_res,
            ipw=ipw_res,
            dr=dr_res,
            diagnostics=diag,
        )

    @staticmethod
    def to_dict(results: CausalResults) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "naive": {
                "ate": results.naive.ate,
                "n": results.naive.n,
                "treated_rate": results.naive.treated_rate,
            },
            "dr": {
                "ate": results.dr.ate,
                "std_error": results.dr.std_error,
                "ci_lower": results.dr.ci_lower,
                "ci_upper": results.dr.ci_upper,
                "n": results.dr.n,
                "overlap_min": results.dr.overlap_min,
                "overlap_max": results.dr.overlap_max,
                "ess": results.dr.ess,
            },
            "diagnostics": {
                "n": results.diagnostics.n,
                "treated_rate": results.diagnostics.treated_rate,
                "overlap_min": results.diagnostics.overlap_min,
                "overlap_max": results.diagnostics.overlap_max,
                "ess": results.diagnostics.ess,
            },
        }

        if results.ipw is not None:
            out["ipw"] = {
                "ate": results.ipw.ate,
                "ate_stabilized": results.ipw.ate_stabilized,
                "std_error": results.ipw.std_error,
                "n": results.ipw.n,
                "overlap_min": results.ipw.overlap_min,
                "overlap_max": results.ipw.overlap_max,
                "treated_rate": results.ipw.treated_rate,
            }
        else:
            out["ipw"] = None

        return out