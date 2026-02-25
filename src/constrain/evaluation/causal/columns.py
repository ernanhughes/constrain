# constrain/evaluation/causal/columns.py
from __future__ import annotations

# Canonical column names for causal evaluation.
# Anything building a causal dataframe must produce these fields.

TREATMENT_ANY = "treatment_any"
TREATMENT_ROLLBACK = "treatment_rollback"
TREATMENT_RESET = "treatment_reset"

OUTCOME_NEXT = "collapse_next"
OUTCOME_H3 = "collapse_within_3"

ACTION = "policy_action"

ENERGY = "total_energy"
SLOPE = "energy_slope"
HARD = "consecutive_hard"
RISING = "consecutive_rising"
ITER = "iteration"
ATTEMPT = "attempt"
TEMP = "temperature"
DELTA_TEMP = "delta_temp"

COLLAPSE_FLAG = "collapse_flag"
RUN_ID = "run_id"
PROBLEM_ID = "problem_id"