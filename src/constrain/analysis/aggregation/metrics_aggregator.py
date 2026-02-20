from __future__ import annotations

from typing import Optional
import pandas as pd
from sklearn import logger
from sklearn.metrics import roc_auc_score
import numpy as np

from constrain.data.memory import Memory
import time


class MetricsAggregator:

    # -------------------------------------------------
    # BUILD RUN DATAFRAME
    # -------------------------------------------------

    @staticmethod
    def build_run_dataframe(memory: Memory, run_id: str) -> pd.DataFrame:
        steps = memory.steps.get_by_run(run_id)
        if not steps:
            raise ValueError(f"No steps found for run_id={run_id}")

        # ✅ Pull ALL data from Step objects (including energy metrics)
        steps_df = pd.DataFrame([{
            "step_id": s.id,
            "run_id": s.run_id,
            "problem_id": s.problem_id,
            "iteration": s.iteration,
            "phase": s.phase,
            "temperature": s.temperature,
            "policy_action": s.policy_action,
            "gold_answer": getattr(s, "gold_answer", None),
            "extracted_answer": getattr(s, "extracted_answer", None),
            # ✅ Energy metrics from Step (not metrics table)
            "total_energy": getattr(s, "total_energy", 0.0),
            "grounding_energy": getattr(s, "grounding_energy", 0.0),
            "stability_energy": getattr(s, "stability_energy", 0.0),
            "accuracy": getattr(s, "accuracy", 0.0),
            "correctness": getattr(s, "correctness", 0.0),
            "phase_value": getattr(s, "phase_value", None),
        } for s in steps])

        # ✅ Still load additional metrics from metrics table (spectral, text signals, etc.)
        step_ids = steps_df["step_id"].tolist()
        metrics_by_step = memory.metrics.get_by_steps(step_ids)

        if metrics_by_step:
            metrics_rows = []
            for step_id, metric_dict in metrics_by_step.items():
                row = {"step_id": step_id}
                row.update(metric_dict)
                metrics_rows.append(row)
            metrics_df = pd.DataFrame(metrics_rows)
            
            # Merge additional metrics
            full_df = steps_df.merge(metrics_df, on="step_id", how="left")
        else:
            full_df = steps_df

        # Ensure numeric
        numeric_cols = [
            "total_energy", "grounding_energy", "stability_energy",
            "accuracy", "correctness", "phase_value",
        ]
        for col in numeric_cols:
            if col in full_df.columns:
                full_df[col] = pd.to_numeric(full_df[col], errors="coerce").fillna(0.0)

        return full_df

    # -------------------------------------------------
    # DUMP CSV
    # -------------------------------------------------

    @staticmethod
    def dump_run_csv(memory: Memory, run_id: str, path: Optional[str] = None):

        df = MetricsAggregator.build_run_dataframe(memory, run_id)

        if path is None:
            path = f"run_{run_id}_metrics.csv"

        df.to_csv(path, index=False)
        print(f"✅ Data written to {path}")

        MetricsAggregator.print_run_summary(df)

        return path

    # -------------------------------------------------
    # SUMMARY
    # -------------------------------------------------

    @staticmethod
    def print_run_summary(df: pd.DataFrame):

        print("\n" + "="*50)
        print("RUN METRIC SUMMARY")
        print("="*50)

        n_steps = len(df)
        n_problems = df["problem_id"].nunique()

        mean_accuracy = df["accuracy"].mean()

        final_accuracy = (
            df.sort_values("iteration")
              .groupby("problem_id")
              .tail(1)["accuracy"]
              .mean()
        )

        mean_energy = df["total_energy"].mean()

        collapse_rate = (
            (df["phase"].isin(["unstable", "collapse"])).mean()
        )

        # Next-step collapse prediction
        df_sorted = df.sort_values(["problem_id", "iteration"])
        df_sorted["collapse_next"] = (
            df_sorted.groupby("problem_id")["phase"]
            .shift(-1)
            .isin(["unstable", "collapse"])
            .astype(int)
        )

        df_clean = df_sorted.dropna(subset=["collapse_next", "total_energy"])

        if len(df_clean) > 10 and df_clean["collapse_next"].nunique() > 1:
            auc = roc_auc_score(
                df_clean["collapse_next"],
                df_clean["total_energy"]
            )
        else:
            auc = np.nan

        print(f"Steps:               {n_steps}")
        print(f"Problems:            {n_problems}")
        print(f"Mean Accuracy:       {mean_accuracy:.4f}")
        print(f"Final Accuracy:      {final_accuracy:.4f}")
        print(f"Mean Energy:         {mean_energy:.4f}")
        print(f"Collapse Rate:       {collapse_rate:.2%}")
        print(f"AUC (energy->collapse): {auc:.4f}")
        print("="*50 + "\n")