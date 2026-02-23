"""constrain.services.policy_evaluation_service
Service for evaluating policy intervention effectiveness.
"""
from __future__ import annotations

import time
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import logging

import numpy as np
import pandas as pd

from constrain.data.memory import Memory
from constrain.data.schemas.policy_evaluation import PolicyEvaluationDTO
from constrain.data.schemas.step import StepDTO
from constrain.config import get_config

logger = logging.getLogger(__name__)


@dataclass
class BootstrapResult:
    """Statistical comparison result with confidence intervals."""
    mean_diff: float
    ci_lower: float
    ci_upper: float
    significant: bool
    n_baseline: int
    n_treatment: int


@dataclass
class RunEvaluationSummary:
    """Aggregated metrics for a single run."""
    run_id: str
    policy_id: int
    seed: Optional[int]
    accuracy: float
    intervention_rate: float
    avg_energy: float
    avg_recursions: float
    intervention_help_rate: float
    intervention_harm_rate: float
    collapse_rate: float
    num_problems: int


class PolicyEvaluationService:
    """
    Evaluates policy intervention effectiveness.
    
    Capabilities:
    - Reconstruct problem trajectories from step logs
    - Compute per-problem outcomes (correctness, intervention effect)
    - Compare policies with statistical significance (bootstrap)
    - Generate reports (console + CSV)
    - Query historical evaluations for comparison
    """

    def __init__(self, memory: Memory):
        self.memory = memory

    # ============================================================
    # Core Evaluation
    # ============================================================

    def evaluate_run(self, run_id: str) -> Tuple[RunEvaluationSummary, pd.DataFrame]:
        """
        Reconstruct and evaluate a single run.
        Returns (summary, problem_dataframe).
        """
        logger.info("Starting policy evaluation for run: %s", run_id)

        # Load run metadata
        run = self.memory.runs.get_by_id(run_id)
        if not run:
            raise ValueError(f"Run not found: {run_id}")

        # Load steps
        steps = self.memory.steps.get_by_run(run_id)
        if not steps:
            raise ValueError(f"No steps found for run: {run_id}")

        # Build problem-level dataframe
        problem_df = self._build_problem_dataframe(run_id, steps, run)

        if problem_df.empty:
            raise ValueError("No problem data reconstructed.")

        # Compute summary
        summary = RunEvaluationSummary(
            run_id=run_id,
            policy_id=run.policy_id,
            seed=run.seed,
            accuracy=float(problem_df["final_correct"].mean()),
            intervention_rate=float(problem_df["any_intervention"].mean()),
            avg_energy=float(problem_df["avg_energy"].mean()),
            avg_recursions=float(problem_df["num_iterations"].mean()),
            intervention_help_rate=float(problem_df["intervention_helped"].mean()),
            intervention_harm_rate=float(problem_df["intervention_harmed"].mean()),
            collapse_rate=float(problem_df["collapsed"].mean()) if "collapsed" in problem_df.columns else 0.0,
            num_problems=len(problem_df),
        )

        # Persist to database
        evaluations = self._dto_list_from_dataframe(problem_df, run)
        if evaluations:
            self.memory.policy_evaluations.bulk_create(evaluations)
            logger.info("Persisted %d policy evaluations for run: %s", len(evaluations), run_id)

        return summary, problem_df

    def _build_problem_dataframe(
        self, run_id: str, steps: List[StepDTO], run
    ) -> pd.DataFrame:
        """Reconstruct problem-level outcomes from step trajectories."""

        # Group steps by problem
        problems: Dict[int, List[StepDTO]] = {}
        for step in steps:
            if step.problem_id not in problems:
                problems[step.problem_id] = []
            problems[step.problem_id].append(step)

        problem_rows = []
        cfg = get_config()

        for problem_id, problem_steps in problems.items():
            # Sort by iteration
            problem_steps.sort(key=lambda s: s.iteration)

            if not problem_steps:
                continue

            # Final correctness
            final_correct = bool(problem_steps[-1].correctness) if problem_steps[-1].correctness is not None else False

            # Intervention flags
            any_intervention = any(s.policy_action in ("REVERT", "RESET") for s in problem_steps)
            num_interventions = sum(1 for s in problem_steps if s.policy_action in ("REVERT", "RESET"))

            # Energy metrics
            energies = [s.total_energy for s in problem_steps]
            avg_energy = sum(energies) / len(energies) if energies else 0.0
            max_energy = max(energies) if energies else 0.0

            # Collapse detection
            collapsed = any(e > cfg.tau_hard for e in energies)

            # Intervention effectiveness
            intervention_helped = False
            intervention_harmed = False

            for i in range(1, len(problem_steps)):
                prev = problem_steps[i - 1]
                curr = problem_steps[i]

                if prev.policy_action in ("REVERT", "RESET"):
                    before_correct = bool(prev.correctness) if prev.correctness is not None else False
                    after_correct = bool(curr.correctness) if curr.correctness is not None else False

                    if not before_correct and after_correct:
                        intervention_helped = True
                    elif before_correct and not after_correct:
                        intervention_harmed = True

            problem_rows.append({
                "problem_id": problem_id,
                "final_correct": final_correct,
                "any_intervention": any_intervention,
                "num_interventions": num_interventions,
                "avg_energy": avg_energy,
                "max_energy": max_energy,
                "num_iterations": len(problem_steps),
                "intervention_helped": intervention_helped,
                "intervention_harmed": intervention_harmed,
                "collapsed": collapsed,
            })

        return pd.DataFrame(problem_rows)

    def _dto_list_from_dataframe(
        self, df: pd.DataFrame, run
    ) -> List[PolicyEvaluationDTO]:
        """Convert problem dataframe to DTOs for persistence."""
        now = time.time()
        evaluations = []

        for _, row in df.iterrows():
            evaluations.append(PolicyEvaluationDTO(
                run_id=run.run_id,
                problem_id=int(row["problem_id"]),
                policy_id=run.policy_id,
                seed=run.seed,
                experiment_id=run.experiment_id,
                final_correct=bool(row["final_correct"]),
                any_intervention=bool(row["any_intervention"]),
                intervention_helped=bool(row["intervention_helped"]),
                intervention_harmed=bool(row["intervention_harmed"]),
                collapsed=bool(row.get("collapsed", False)),
                num_iterations=int(row["num_iterations"]),
                avg_energy=float(row["avg_energy"]),
                max_energy=float(row["max_energy"]),
                tau_soft=run.tau_soft,
                tau_medium=run.tau_medium,
                tau_hard=run.tau_hard,
                created_at=now,
            ))

        return evaluations

    # ============================================================
    # Statistical Comparison
    # ============================================================

    def bootstrap_diff(
        self,
        baseline: List[float],
        treatment: List[float],
        n_iterations: int = 2000,
        seed: int = 42,
    ) -> BootstrapResult:
        """Compute bootstrap confidence interval for accuracy difference."""
        rng = np.random.RandomState(seed)

        if len(baseline) == 0 or len(treatment) == 0:
            return BootstrapResult(
                mean_diff=np.nan, ci_lower=np.nan, ci_upper=np.nan,
                significant=False, n_baseline=len(baseline), n_treatment=len(treatment)
            )

        diffs = []
        for _ in range(n_iterations):
            idx_a = rng.choice(len(baseline), len(baseline), replace=True)
            idx_b = rng.choice(len(treatment), len(treatment), replace=True)
            diffs.append(np.array(baseline)[idx_a].mean() - np.array(treatment)[idx_b].mean())

        lower = np.percentile(diffs, 2.5)
        upper = np.percentile(diffs, 97.5)
        mean_diff = np.mean(diffs)

        return BootstrapResult(
            mean_diff=float(mean_diff),
            ci_lower=float(lower),
            ci_upper=float(upper),
            significant=(lower > 0 or upper < 0),  # CI doesn't cross zero
            n_baseline=len(baseline),
            n_treatment=len(treatment),
        )

    def compare_policies(
        self,
        policy_ids: List[int],
        experiment_id: Optional[int] = None,
    ) -> Dict[int, BootstrapResult]:
        """Compare all policies vs baseline (policy 0)."""
        # Fetch all evaluations
        by_policy: Dict[int, List[float]] = {}

        for pid in policy_ids:
            evals = self.memory.policy_evaluations.get_by_policy_id(
                policy_id=pid, experiment_id=experiment_id
            )
            by_policy[pid] = [float(e.final_correct) for e in evals]

        # Compare each vs baseline
        baseline = by_policy.get(0, [])
        results = {}

        for pid in policy_ids:
            if pid == 0:
                continue
            treatment = by_policy.get(pid, [])
            results[pid] = self.bootstrap_diff(baseline, treatment)

        return results

    # ============================================================
    # Reporting
    # ============================================================

    def generate_report(self, run_id: str) -> str:
        """Generate human-readable console report with historical comparison."""

        evaluations = self.memory.policy_evaluations.get_by_run_id(run_id)
        if not evaluations:
            return "No policy evaluations found for this run."

        df = pd.DataFrame([e.model_dump() for e in evaluations])

        # Current run metrics
        accuracy = df["final_correct"].mean() * 100
        intervention_rate = df["any_intervention"].mean() * 100
        helped = int(df["intervention_helped"].sum())
        harmed = int(df["intervention_harmed"].sum())
        collapsed = int(df["collapsed"].sum()) if "collapsed" in df.columns else 0
        avg_energy = df["avg_energy"].mean()
        avg_iterations = df["num_iterations"].mean()

        # Thresholds
        tau_soft = df["tau_soft"].iloc[0] if len(df) > 0 else 0.0
        tau_medium = df["tau_medium"].iloc[0] if len(df) > 0 else 0.0
        tau_hard = df["tau_hard"].iloc[0] if len(df) > 0 else 0.0

        # Historical comparison
        current_policy_id = int(df["policy_id"].iloc[0]) if len(df) > 0 else None
        historical = None

        if current_policy_id is not None:
            hist_evals = self.memory.policy_evaluations.get_by_policy_id(
                policy_id=current_policy_id
            )
            hist_evals = [e for e in hist_evals if e.run_id != run_id]

            if hist_evals:
                hist_df = pd.DataFrame([e.model_dump() for e in hist_evals])
                historical = {
                    "n_runs": hist_df["run_id"].nunique(),
                    "n_problems": len(hist_df),
                    "avg_accuracy": hist_df["final_correct"].mean() * 100,
                    "avg_intervention_rate": hist_df["any_intervention"].mean() * 100,
                    "total_helped": int(hist_df["intervention_helped"].sum()),
                    "total_harmed": int(hist_df["intervention_harmed"].sum()),
                }

        # Build report
        def fmt_delta(current: float, historical: float) -> str:
            delta = current - historical
            sign = "+" if delta >= 0 else ""
            return f"{sign}{delta:.2f}%"

        lines = [
            "╔══════════════════════════════════════════════════════════════╗",
            "║           POLICY EVALUATION REPORT                           ║",
            "╠══════════════════════════════════════════════════════════════╣",
            f"║ Run ID: {run_id:<50} ║",
            f"║ Policy ID: {current_policy_id:<45} ║",
            "╠══════════════════════════════════════════════════════════════╣",
            "║ CURRENT RUN METRICS                                          ║",
            "╠──────────────────────────────────────────────────────────────╣",
            f"║ Accuracy:              {accuracy:>6.2f}%                           ║",
            f"║ Intervention Rate:     {intervention_rate:>6.2f}%            ║",
            f"║ Interventions Helped:  {helped:>6d}                          ║",
            f"║ Interventions Harmed:  {harmed:>6d}                          ║",
            f"║ Collapses:             {collapsed:>6d}                          ║",
            f"║ Avg Energy:            {avg_energy:>6.4f}                    ║",
            f"║ Avg Iterations:        {avg_iterations:>6.2f}                ║",
            "╠──────────────────────────────────────────────────────────────╣",
            "║ THRESHOLDS USED                                              ║",
            "╠──────────────────────────────────────────────────────────────╣",
            f"║ Tau Soft:   {tau_soft:>6.4f}                                 ║",
            f"║ Tau Medium: {tau_medium:>6.4f}                               ║",
            f"║ Tau Hard:   {tau_hard:>6.4f}                                 ║",
        ]

        if historical:
            hist = historical
            acc_delta = fmt_delta(accuracy, hist["avg_accuracy"])
            int_delta = fmt_delta(intervention_rate, hist["avg_intervention_rate"])

            if accuracy > hist["avg_accuracy"] and harmed <= hist["total_harmed"] / max(1, hist["n_runs"]):
                verdict = "✅ OUTPERFORMS HISTORY"
            elif accuracy < hist["avg_accuracy"] and harmed > hist["total_harmed"] / max(1, hist["n_runs"]):
                verdict = "❌ UNDERPERFORMS HISTORY"
            else:
                verdict = "⚠️  COMPARABLE TO HISTORY"

            lines.extend([
                "╠══════════════════════════════════════════════════════════════╣",
                "║ COMPARISON TO PREVIOUS RUNS (Same Policy)                    ║",
                "╠──────────────────────────────────────────────────────────────╣",
                f"║ Historical Runs:       {hist['n_runs']:>6d}                          ║",
                f"║ Historical Problems:   {hist['n_problems']:>6d}                          ║",
                f"║ Historical Accuracy:   {hist['avg_accuracy']:>6.2f}%                     ║",
                f"║ Historical Interv. Rate: {hist['avg_intervention_rate']:>6.2f}%            ║",
                f"║ Historical Helped:     {hist['total_helped']:>6d}                          ║",
                f"║ Historical Harmed:     {hist['total_harmed']:>6d}                          ║",
                "╠──────────────────────────────────────────────────────────────╣",
                f"║ Δ Accuracy:            {acc_delta:>6s}                     ║",
                f"║ Δ Intervention Rate:   {int_delta:>6s}                     ║",
                "╠──────────────────────────────────────────────────────────────╣",
                f"║ VERDICT: {verdict:<41} ║",
            ])
        else:
            lines.extend([
                "╠══════════════════════════════════════════════════════════════╣",
                "║ COMPARISON TO PREVIOUS RUNS                                  ║",
                "╠──────────────────────────────────────────────────────────────╣",
                "║ No historical data available for this policy.                ║",
                "║ Run more evaluations to enable comparison.                   ║",
            ])

        lines.append("╚══════════════════════════════════════════════════════════════╝")
        return "\n".join(lines)

    def export_csv(self, run_ids: List[str], output_path: str):
        """Export evaluation results to CSV for external analysis."""
        all_evals = []
        for run_id in run_ids:
            all_evals.extend(self.memory.policy_evaluations.get_by_run_id(run_id))

        if not all_evals:
            logger.warning("No evaluations found for export")
            return

        df = pd.DataFrame([e.model_dump() for e in all_evals])
        df.to_csv(output_path, index=False)
        logger.info("Exported %d evaluations to %s", len(all_evals), output_path)