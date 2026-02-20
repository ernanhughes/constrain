# constrain/analysis/metrics_aggregator.py

from __future__ import annotations

from typing import Optional

import pandas as pd

from constrain.data.memory import Memory


class MetricsAggregator:

    @staticmethod
    def build_run_dataframe(memory: Memory, run_id: str) -> pd.DataFrame:
        """
        Build a wide dataframe for a given run.

        Returns:
            DataFrame with:
                - Step metadata columns
                - One column per metric_name
        """

        # -----------------------------
        # Load steps
        # -----------------------------

        steps = memory.steps.list(
            limit=None,
            filters={"run_id": run_id},
            desc=False,
        )

        if not steps:
            raise ValueError(f"No steps found for run_id={run_id}")

        steps_df = pd.DataFrame([{
            "step_id": s.id,
            "run_id": s.run_id,
            "problem_id": s.problem_id,
            "iteration": s.iteration,
            "phase": s.phase,
            "total_energy": s.total_energy,
            "grounding_energy": s.grounding_energy,
            "stability_energy": s.stability_energy,
            "temperature": s.temperature,
            "policy_action": s.policy_action,
            "accuracy": s.accuracy,
            "correctness": s.correctness,
        } for s in steps])

        # -----------------------------
        # Load metrics
        # -----------------------------

        metrics = memory.metrics.list(
            limit=None,
            filters=None,  # we'll filter manually
            desc=False,
        )

        metrics = [m for m in metrics if m.step_id in set(steps_df["step_id"])]

        metrics_df = pd.DataFrame([{
            "step_id": m.step_id,
            "metric_name": m.metric_name,
            "metric_value": m.metric_value,
            "stage": m.stage,
        } for m in metrics])

        if metrics_df.empty:
            return steps_df

        # -----------------------------
        # Pivot to wide format
        # -----------------------------

        metrics_wide = (
            metrics_df
            .pivot_table(
                index="step_id",
                columns="metric_name",
                values="metric_value",
                aggfunc="first"
            )
            .reset_index()
        )

        # Flatten multi-index if needed
        metrics_wide.columns.name = None

        # -----------------------------
        # Merge
        # -----------------------------

        full_df = steps_df.merge(metrics_wide, on="step_id", how="left")

        return full_df


    @staticmethod
    def dump_run_csv(memory: Memory, run_id: str, path: Optional[str] = None):
        """
        Dump aggregated run metrics to CSV.
        """

        df = MetricsAggregator.build_run_dataframe(memory, run_id)

        if path is None:
            path = f"run_{run_id}_metrics.csv"

        df.to_csv(path, index=False)
        print(f"✅ Metrics CSV written to {path}")

        return path
