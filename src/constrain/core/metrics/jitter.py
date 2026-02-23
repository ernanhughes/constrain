from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
import numpy as np
import torch


@dataclass
class JitterMetrics:
    mean_jitter: float
    std_jitter: float
    max_jitter: float
    p95_jitter: float
    total_steps: int


class JitterTracker:
    """
    Tracks parameter thrash (step-to-step L2 change) for interpretability.

    Accepts either:
      - dict-like params
      - objects with to_dict()
    """

    def __init__(self, param_names: Optional[List[str]] = None):
        self.param_names = param_names or ["tau_hard", "temperature", "recursion_limit"]
        self.jitter_history: List[float] = []
        self.last_params: Optional[Any] = None

    def _to_dict(self, params: Any) -> Dict[str, Any]:
        if hasattr(params, "to_dict"):
            return params.to_dict()
        if isinstance(params, dict):
            return params
        # best-effort: dataclass-like
        return getattr(params, "__dict__", {})

    def _to_vector(self, params: Any) -> torch.Tensor:
        d = self._to_dict(params)
        vals = []
        for k in self.param_names:
            if k in d and d[k] is not None:
                vals.append(float(d[k]))
        return torch.tensor(vals, dtype=torch.float32)

    def record(self, params: Any) -> float:
        cur = self._to_vector(params)
        if self.last_params is None:
            self.last_params = params
            return 0.0
        prev = self._to_vector(self.last_params)
        j = torch.norm(cur - prev).item()
        self.jitter_history.append(float(j))
        self.last_params = params
        return float(j)

    def compute_metrics(self) -> JitterMetrics:
        if not self.jitter_history:
            return JitterMetrics(0.0, 0.0, 0.0, 0.0, 0)
        arr = np.array(self.jitter_history, dtype=float)
        return JitterMetrics(
            mean_jitter=float(arr.mean()),
            std_jitter=float(arr.std()),
            max_jitter=float(arr.max()),
            p95_jitter=float(np.percentile(arr, 95)),
            total_steps=int(arr.size),
        )

    def reset(self):
        self.jitter_history = []
        self.last_params = None
