"""constrain.services.intervention_timing_service
Intervention Timing Analysis Service

Determines whether interventions occur:
- Too late (collapse at t+1)
- Too early
- At optimal margin
- Or randomly

Returns distributional statistics, not just prints.
"""

from typing import Dict

import numpy as np
import pandas as pd

from constrain.config import get_config
from constrain.data.memory import Memory


class InterventionTimingService:
    def __init__(self, memory: Memory):
        self.memory = memory
        self.cfg = get_config()

    def analyze_run(self, run_id: str) -> Dict:
        """
        Analyze intervention timing relative to collapse events.

        Returns:
            {
                "mean_delta": float,
                "median_delta": float,
                "pct_delta_le_1": float,
                "pct_delta_le_2": float,
                "pct_delta_ge_3": float,
                "n_interventions": int,
                "n_collapsed_problems": int,
                "histogram": Dict[int, int],
                "verdict": str
            }
        """
        # Load steps
        steps = self.memory.steps.get_by_run(run_id)
        if not steps:
            return self._empty_result("no_steps")

        # Convert to DataFrame
        df = pd.DataFrame([s.model_dump() for s in steps])
        df = df.sort_values(["problem_id", "iteration"]).reset_index(drop=True)

        # Identify collapse iterations per problem
        tau_hard = self.cfg.tau_hard
        df["is_collapse"] = df["total_energy"] > tau_hard

        collapse_iters = (
            df[df["is_collapse"]]
            .groupby("problem_id")["iteration"]
            .min()
            .to_dict()
        )

        # Identify interventions
        interventions = df[df["policy_action"] != "ACCEPT"][
            ["problem_id", "iteration", "policy_action"]
        ].copy()

        if len(interventions) == 0:
            return self._empty_result("no_interventions")

        # Compute delta for each intervention
        deltas = []
        for _, row in interventions.iterrows():
            pid = row["problem_id"]
            iter_idx = row["iteration"]

            if pid not in collapse_iters:
                continue  # No collapse in this problem

            collapse_iter = collapse_iters[pid]

            if iter_idx >= collapse_iter:
                continue  # Intervention after collapse

            delta = collapse_iter - iter_idx
            deltas.append(delta)

        if len(deltas) == 0:
            return self._empty_result("no_pre_collapse_interventions")

        deltas = np.array(deltas)

        # Compute statistics
        histogram = {int(k): int(v) for k, v in zip(*np.histogram(deltas, bins=range(0, 11))[::-1])}

        result = {
            "mean_delta": float(np.mean(deltas)),
            "median_delta": float(np.median(deltas)),
            "std_delta": float(np.std(deltas)),
            "pct_delta_le_1": float(np.mean(deltas <= 1) * 100),
            "pct_delta_le_2": float(np.mean(deltas <= 2) * 100),
            "pct_delta_ge_3": float(np.mean(deltas >= 3) * 100),
            "n_interventions": len(interventions),
            "n_collapsed_problems": len(collapse_iters),
            "n_analyzed_interventions": len(deltas),
            "histogram": histogram,
        }

        # Auto verdict
        result["verdict"] = self._compute_verdict(result)

        return result

    def _compute_verdict(self, stats: Dict) -> str:
        if stats["pct_delta_le_1"] > 50:
            return "⚠️  INTERVENING TOO LATE (>50% within 1 step of collapse)"
        elif stats["pct_delta_ge_3"] > 40:
            return "✅ INTERVENING EARLY ENOUGH (>40% with ≥3 steps margin)"
        elif stats["median_delta"] >= 2:
            return "⚠️  TIMING MIXED (median 2 steps)"
        else:
            return "⚠️  TIMING SUBOPTIMAL (median <2 steps)"

    def _empty_result(self, reason: str) -> Dict:
        return {
            "mean_delta": None,
            "median_delta": None,
            "pct_delta_le_1": None,
            "pct_delta_le_2": None,
            "pct_delta_ge_3": None,
            "n_interventions": 0,
            "n_collapsed_problems": 0,
            "n_analyzed_interventions": 0,
            "histogram": {},
            "verdict": f"SKIPPED: {reason}",
        }


def main():
    from constrain.config import get_config
    from constrain.data.memory import Memory

    cfg = get_config()
    memory = Memory(cfg.db_url)

    # Get most recent run
    runs = memory.runs.get_recent(limit=1)
    if not runs:
        print("❌ No runs found")
        return

    run_id = runs[0].run_id
    print(f"Analyzing run: {run_id}")

    service = InterventionTimingService(memory)
    result = service.analyze_run(run_id)

    print("\n" + "=" * 60)
    print("INTERVENTION TIMING ANALYSIS")
    print("=" * 60)
    print(f"Verdict: {result['verdict']}")
    print(f"\nInterventions analyzed: {result['n_analyzed_interventions']}")
    print(f"Collapsed problems: {result['n_collapsed_problems']}")
    print("\nTiming statistics:")
    print(f"  Mean delta: {result['mean_delta']:.2f} steps")
    print(f"  Median delta: {result['median_delta']:.2f} steps")
    print(f"  Std delta: {result.get('std_delta', 'N/A')}")
    print("\nDistribution:")
    print(f"  ≤1 step to collapse: {result['pct_delta_le_1']:.1f}%")
    print(f"  ≤2 steps to collapse: {result['pct_delta_le_2']:.1f}%")
    print(f"  ≥3 steps to collapse: {result['pct_delta_ge_3']:.1f}%")
    print("=" * 60)


if __name__ == "__main__":
    main()