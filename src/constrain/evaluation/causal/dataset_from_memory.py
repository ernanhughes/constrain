# constrain/evaluation/causal/dataset_from_memory.py
from __future__ import annotations

import pandas as pd

from constrain.data.memory import Memory

from . import columns as C


def build_causal_df_from_memory(memory: Memory, run_id: str) -> pd.DataFrame:
    """
    Build canonical causal dataframe for a run by merging:
      - steps (StepDTO dumps)
      - reasoning_state_snapshots (ReasoningStateSnapshotDTO dumps)

    Expects both contain at least:
      run_id, problem_id, iteration

    Snapshot is expected (in your new control system) to contain:
      attempt, policy_action, collapse_flag, energy_slope, counters
    """
    steps = memory.steps.get_by_run(run_id)
    snaps = memory.reasoning_state_snapshots.get_by_run(run_id)

    if not steps or not snaps:
        return pd.DataFrame()

    steps_df = pd.DataFrame([s.model_dump() for s in steps])
    snap_df = pd.DataFrame([s.model_dump() for s in snaps])

    # Merge on canonical keys
    df = steps_df.merge(
        snap_df,
        on=[C.RUN_ID, C.PROBLEM_ID, C.ITER],
        how="inner",
        suffixes=("_step", "_snap"),
    )

    # --- Treatments (from policy action) ---
    if C.ACTION in df.columns:
        df[C.TREATMENT_ANY] = (df[C.ACTION] != "ACCEPT").astype(int)
        df[C.TREATMENT_ROLLBACK] = (df[C.ACTION] == "ROLLBACK").astype(int)
        df[C.TREATMENT_RESET] = (df[C.ACTION] == "RESET").astype(int)
    else:
        # fallback if old naming used
        df[C.TREATMENT_ANY] = 0
        df[C.TREATMENT_ROLLBACK] = 0
        df[C.TREATMENT_RESET] = 0

    # --- Delta temperature ---
    if C.TEMP in df.columns:
        df[C.DELTA_TEMP] = (
            df[C.TEMP] - df.groupby(C.PROBLEM_ID)[C.TEMP].shift(1)
        ).fillna(0.0)
    else:
        df[C.DELTA_TEMP] = 0.0

    # --- Outcomes: collapse next / within horizon ---
    # Prefer ordering by attempt (true "model call" time). Fallback to iteration.
    sort_cols = [C.PROBLEM_ID, C.ATTEMPT] if C.ATTEMPT in df.columns else [C.PROBLEM_ID, C.ITER]
    df = df.sort_values(sort_cols)

    if C.COLLAPSE_FLAG in df.columns:
        df[C.OUTCOME_NEXT] = (
            df.groupby(C.PROBLEM_ID)[C.COLLAPSE_FLAG]
            .shift(-1)
            .fillna(0)
            .astype(int)
        )

        # Horizon=3
        df[C.OUTCOME_H3] = (
            df.groupby(C.PROBLEM_ID)[C.COLLAPSE_FLAG]
            .rolling(3)
            .max()
            .reset_index(level=0, drop=True)
            .shift(-1)
            .fillna(0)
            .astype(int)
        )
    else:
        df[C.OUTCOME_NEXT] = 0
        df[C.OUTCOME_H3] = 0

    return df