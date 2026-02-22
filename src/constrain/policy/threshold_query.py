from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Sequence


@dataclass(frozen=True)
class ThresholdQuery:
    """
    Defines what data the DB threshold provider will use.
    All fields are logs-only. No recomputation.
    """
    # Scope
    run_id: Optional[str] = None                 # use this run only
    include_run_ids: Optional[Sequence[str]] = None  # explicit set of runs
    last_n_runs: Optional[int] = None            # most recent runs (by start_time)
    last_n_steps: Optional[int] = None           # most recent steps (by timestamp)

    # Filters
    exclude_policy_ids: Optional[Sequence[int]] = None
    require_correctness_nonnull: bool = False    # optional

    # Quantiles
    q_soft: float = 0.80
    q_medium: float = 0.90
    q_hard: float = 0.97

    # Safety
    min_samples: int = 20                       # avoid tiny-n collapse