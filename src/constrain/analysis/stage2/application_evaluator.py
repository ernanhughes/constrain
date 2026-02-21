from __future__ import annotations

import pandas as pd

from constrain.analysis.aggregation.metrics_aggregator import MetricsAggregator
from constrain.config import get_config


class ApplicationEvaluator:

    def __init__(self, memory):
        self.memory = memory

    # ============================================================
    # Public API
    # ============================================================

    def evaluate_run(self, run_id: str):

        df = self._build_problem_dataframe(run_id)

        if df.empty:
            raise ValueError("No problem data reconstructed.")

        summary = {
            "run_id": run_id,
            "accuracy": float(df["final_correct"].mean()),
            "intervention_rate": float(df["any_intervention"].mean()),
            "avg_energy": float(df["avg_energy"].mean()),
            "avg_recursions": float(df["num_iterations"].mean()),
            "intervention_help_rate": float(df["intervention_helped"].mean()),
            "intervention_harm_rate": float(df["intervention_harmed"].mean()),
            "num_problems": int(len(df)),
        }

        return summary, df

    # ============================================================
    # Core Reconstruction Logic
    # ============================================================

    def _build_problem_dataframe(self, run_id: str):

        steps_df = self._load_steps(run_id)

        if steps_df.empty:
            return pd.DataFrame()

        problem_rows = []

        for problem_id, group in steps_df.groupby("problem_id"):

            group = group.sort_values("iteration").reset_index(drop=True)

            final_row = group.iloc[-1]
            final_correct = self._is_correct(final_row)

            any_intervention = (group["policy_action"] != "ACCEPT").any()
            num_interventions = int(
                (group["policy_action"] != "ACCEPT").sum()
            )

            avg_energy = float(group["total_energy"].mean(skipna=True))
            max_energy = float(group["total_energy"].max(skipna=True))

            num_iterations = int(len(group))

            # ----------------------------------------------------
            # Intervention Effectiveness (corrected logic)
            # ----------------------------------------------------

            intervention_helped = False
            intervention_harmed = False

            tau_hard = get_config().tau_hard
            collapsed = (group["total_energy"] > tau_hard).any()
            for i in range(1, len(group)):

                prev = group.iloc[i - 1]
                curr = group.iloc[i]



                # Intervention triggered at previous step
                if prev["policy_action"] != "ACCEPT":

                    before_correct = self._is_correct(prev)
                    after_correct = self._is_correct(curr)

                    if not before_correct and after_correct:
                        intervention_helped = True

                    if before_correct and not after_correct:
                        intervention_harmed = True

            problem_rows.append(
                {
                    "problem_id": problem_id,
                    "final_correct": final_correct,
                    "any_intervention": bool(any_intervention),
                    "num_interventions": num_interventions,
                    "avg_energy": avg_energy,
                    "max_energy": max_energy,
                    "num_iterations": num_iterations,
                    "intervention_helped": intervention_helped,
                    "intervention_harmed": intervention_harmed,
                    "collapsed": bool(collapsed),   # 🔥 add this
                }
            )

        return pd.DataFrame(problem_rows)

    # ============================================================
    # Helpers
    # ============================================================

    def _load_steps(self, run_id: str):

        df = MetricsAggregator.build_run_dataframe(
            self.memory,
            run_id,
        )

        if df is None or df.empty:
            raise ValueError(f"No steps found for run_id={run_id}")

        required = [
            "problem_id",
            "iteration",
            "policy_action",
            "total_energy",
            "extracted_answer",
            "gold_answer",
        ]

        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        return df

    def _is_correct(self, row):

        # Prefer stored correctness if present
        if "accuracy" in row and not pd.isna(row["accuracy"]):
            return bool(row["accuracy"])

        extracted = str(row.get("extracted_answer", "")).strip()
        gold = str(row.get("gold_answer", "")).strip()

        if not extracted or not gold:
            return False

        return extracted == gold
