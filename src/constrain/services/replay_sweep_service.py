# constrain/services/replay_sweep_service.py

from __future__ import annotations

import itertools
from typing import List, Iterable, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from constrain.data.memory import Memory
from constrain.services.policy_replay_service import PolicyReplayService


class ReplaySweepService:

    # ============================================================
    # PUBLIC ENTRYPOINT
    # ============================================================

    @staticmethod
    def sweep(
        memory: Memory,
        policy_ids: Iterable[int],
        tau_soft_values: Iterable[float],
        tau_medium_values: Iterable[float],
        tau_hard_values: Iterable[float],
        collapse_threshold: Optional[float] = None,
        limit_runs: Optional[int] = None,
        generate_plots: bool = True,
    ) -> pd.DataFrame:

        run_ids = ReplaySweepService._get_all_run_ids(memory)

        if limit_runs:
            run_ids = run_ids[:limit_runs]

        results = []

        for run_id in run_ids:
            print(f"Processing run: {run_id}")

            for policy_id, tau_soft, tau_medium, tau_hard in itertools.product(
                policy_ids,
                tau_soft_values,
                tau_medium_values,
                tau_hard_values,
            ):
                if not (tau_soft <= tau_medium <= tau_hard):
                    continue

                replay = PolicyReplayService.replay_run(
                    memory=memory,
                    run_id=run_id,
                    policy_id=policy_id,
                    tau_soft=tau_soft,
                    tau_medium=tau_medium,
                    tau_hard=tau_hard,
                )

                if replay["total_steps"] == 0:
                    continue

                # Optional collapse calculation
                collapse_rate = None
                if collapse_threshold is not None:
                    collapse_rate = ReplaySweepService._compute_collapse_rate(
                        memory, run_id, collapse_threshold
                    )

                results.append(
                    {
                        **replay,
                        "collapse_rate": collapse_rate,
                    }
                )

        df = pd.DataFrame(results)

        if generate_plots:
            ReplaySweepService._generate_plots(df)

        return df

    # ============================================================
    # UTILITIES
    # ============================================================

    @staticmethod
    def _get_all_run_ids(memory: Memory) -> List[str]:
        runs = memory.runs.list(limit=100000)
        return [r.run_id for r in runs]

    @staticmethod
    def _compute_collapse_rate(
        memory: Memory,
        run_id: str,
        collapse_threshold: float,
    ) -> float:

        steps = memory.steps.list(
            filters={"run_id": run_id},
            order_by="iteration",
            desc=False,
            limit=100000,
        )

        if not steps:
            return 0.0

        energies = [s.total_energy for s in steps]
        collapse_count = sum(e > collapse_threshold for e in energies)

        return collapse_count / len(energies)

    # ============================================================
    # PLOTTING
    # ============================================================

    @staticmethod
    def _generate_plots(df: pd.DataFrame):

        if df.empty:
            return

        print("Generating plots...")

        # 1️⃣ Accept rate vs tau_medium
        plt.figure()
        for policy_id in df["policy_id"].unique():
            subset = df[df["policy_id"] == policy_id]
            grouped = subset.groupby("tau_medium")["accept_rate"].mean()
            plt.plot(grouped.index, grouped.values, label=f"Policy {policy_id}")

        plt.xlabel("tau_medium")
        plt.ylabel("Mean Accept Rate")
        plt.title("Accept Rate vs Tau Medium")
        plt.legend()
        plt.tight_layout()
        plt.show()

        # 2️⃣ Intervention rate vs tau_medium
        plt.figure()
        for policy_id in df["policy_id"].unique():
            subset = df[df["policy_id"] == policy_id]
            grouped = subset.groupby("tau_medium")["intervention_rate"].mean()
            plt.plot(grouped.index, grouped.values, label=f"Policy {policy_id}")

        plt.xlabel("tau_medium")
        plt.ylabel("Mean Intervention Rate")
        plt.title("Intervention Rate vs Tau Medium")
        plt.legend()
        plt.tight_layout()
        plt.show()

        # 3️⃣ Collapse rate vs tau_medium (if available)
        if "collapse_rate" in df.columns and df["collapse_rate"].notnull().any():
            plt.figure()
            for policy_id in df["policy_id"].unique():
                subset = df[df["policy_id"] == policy_id]
                grouped = subset.groupby("tau_medium")["collapse_rate"].mean()
                plt.plot(grouped.index, grouped.values, label=f"Policy {policy_id}")

            plt.xlabel("tau_medium")
            plt.ylabel("Mean Collapse Rate")
            plt.title("Collapse Rate vs Tau Medium")
            plt.legend()
            plt.tight_layout()
            plt.show()

        # 4️⃣ Energy distribution
        plt.figure()
        plt.hist(df["energy_mean"], bins=30)
        plt.xlabel("Mean Energy Per Run")
        plt.ylabel("Frequency")
        plt.title("Energy Distribution Across Runs")
        plt.tight_layout()
        plt.show()



def main():

    memory = Memory()

    df = ReplaySweepService.sweep(
        memory=memory,
        policy_ids=[0, 3, 5],
        tau_soft_values=[0.25],
        tau_medium_values=np.linspace(0.3, 0.7, 10),
        tau_hard_values=[0.8],
        collapse_threshold=0.75,
        limit_runs=200,
    )

    df.to_csv("replay_sweep_results.csv", index=False)
    print("Saved results to replay_sweep_results.csv")


if __name__ == "__main__":
    main()