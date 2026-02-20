# analysis/application_evaluator.py

from __future__ import annotations
import pandas as pd


class ApplicationEvaluator:

    def __init__(self, memory):
        self.memory = memory

    # ============================================================
    # Public API
    # ============================================================

    def evaluate_run(self, run_id: str):

        df = self._build_problem_dataframe(run_id)

        summary = {
            "run_id": run_id,
            "accuracy": df["final_correct"].mean(),
            "intervention_rate": df["any_intervention"].mean(),
            "avg_energy": df["avg_energy"].mean(),
            "avg_recursions": df["num_iterations"].mean(),
            "intervention_help_rate": df["intervention_helped"].mean(),
            "intervention_harm_rate": df["intervention_harmed"].mean(),
        }

        return summary, df

    # ============================================================
    # Core Reconstruction Logic
    # ============================================================

    def _build_problem_dataframe(self, run_id: str):

        steps_df = self._load_steps(run_id)

        problem_rows = []

        for problem_id, group in steps_df.groupby("problem_id"):

            group = group.sort_values("iteration")

            final_row = group.iloc[-1]

            final_correct = self._is_correct(final_row)

            any_intervention = (group["policy_action"] != "ACCEPT").any()
            num_interventions = (group["policy_action"] != "ACCEPT").sum()

            avg_energy = group["total_energy"].mean()
            max_energy = group["total_energy"].max()

            num_iterations = group["iteration"].max() + 1

            # ----------------------------------------------------
            # Intervention Effectiveness
            # ----------------------------------------------------

            intervention_helped = False
            intervention_harmed = False

            for i in range(1, len(group)):
                prev = group.iloc[i - 1]
                curr = group.iloc[i]

                if curr["policy_action"] != "ACCEPT":

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
                    "any_intervention": any_intervention,
                    "num_interventions": num_interventions,
                    "avg_energy": avg_energy,
                    "max_energy": max_energy,
                    "num_iterations": num_iterations,
                    "intervention_helped": intervention_helped,
                    "intervention_harmed": intervention_harmed,
                }
            )

        return pd.DataFrame(problem_rows)

    # ============================================================
    # Helpers
    # ============================================================

    def _load_steps(self, run_id):
        run_id = "run_18cdc06e"

        steps = self.memory.steps.get_by_run(run_id)

        df = pd.DataFrame([s.dict() for s in steps])

        return df

    def _is_correct(self, row):

        extracted = str(row["extracted_answer"]).strip()
        gold = str(row["gold_answer"]).strip()

        return extracted == gold