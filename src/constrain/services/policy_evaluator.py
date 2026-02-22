from __future__ import annotations
import time
from dataclasses import dataclass
from typing import List, Dict, Optional
import logging

from constrain.data.memory import Memory
from constrain.data.schemas.policy_evaluation import PolicyEvaluationDTO
from constrain.data.schemas.step import StepDTO

logger = logging.getLogger(__name__)


class PolicyEvaluator:
    """
    Reconstructs problem trajectories from persisted logs and computes
    Stage 2 evaluation metrics. Stores results for later analysis.
    
    Operates strictly on persisted logs — no recomputation.
    """

    def __init__(self, memory: Memory):
        self.memory = memory

    def evaluate_run(self, run_id: str) -> List[PolicyEvaluationDTO]:
        """
        Evaluate a single run and persist results.
        Returns list of evaluation DTOs.
        """
        logger.info("Starting Stage 2 evaluation for run: %s", run_id)

        # 1. Load run metadata
        run = self.memory.runs.get_by_id(run_id)
        if run is None:
            raise ValueError(f"Run not found: {run_id}")

        # 2. Load all steps for this run
        steps = self.memory.steps.get_by_run(run_id)
        if not steps:
            raise ValueError(f"No steps found for run: {run_id}")

        # 3. Group steps by problem
        problems: Dict[int, List[StepDTO]] = {}
        for step in steps:
            if step.problem_id not in problems:
                problems[step.problem_id] = []
            problems[step.problem_id].append(step)

        # 4. Load policy events for threshold data
        policy_events = self.memory.policy_events.get_by_run_id(run_id)
        thresholds_by_step: Dict[int, Dict[str, float]] = {}
        for event in policy_events:
            thresholds_by_step[event.step_id] = {
                "tau_soft": event.tau_soft,
                "tau_medium": event.tau_medium,
                "tau_hard": event.tau_hard,
            }

        # 5. Compute outcomes per problem
        evaluations: List[PolicyEvaluationDTO] = []
        now = time.time()

        for problem_id, problem_steps in problems.items():
            # Sort by iteration to ensure trajectory order
            problem_steps.sort(key=lambda s: s.iteration)

            if not problem_steps:
                continue

            # Get thresholds from first step's policy event (or fallback to run config)
            first_step = problem_steps[0]
            thresholds = thresholds_by_step.get(first_step.id, {
                "tau_soft": run.tau_soft,
                "tau_medium": run.tau_medium,
                "tau_hard": run.tau_hard,
            })

            # Compute metrics
            outcome = self._compute_problem_outcome(
                run_id=run_id,
                problem_id=problem_id,
                steps=problem_steps,
                policy_id=run.policy_id,
                seed=run.seed,
                experiment_id=run.experiment_id,
                thresholds=thresholds,
                created_at=now,
            )
            evaluations.append(outcome)

        # 6. Persist evaluations
        if evaluations:
            self.memory.policy_evaluations.bulk_create(evaluations)
            logger.info("Persisted %d Stage 2 evaluations for run: %s", len(evaluations), run_id)

        return evaluations

    def _compute_problem_outcome(
        self,
        *,
        run_id: str,
        problem_id: int,
        steps: List[StepDTO],
        policy_id: int,
        seed: Optional[int],
        experiment_id: Optional[int],
        thresholds: Dict[str, float],
        created_at: float,
    ) -> PolicyEvaluationDTO:
        """Compute evaluation metrics for a single problem trajectory."""

        # Final correctness (last iteration)
        final_correct = bool(steps[-1].correctness) if steps[-1].correctness is not None else False

        # Any intervention occurred?
        any_intervention = any(s.policy_action in ("REVERT", "RESET") for s in steps)

        # Intervention effect analysis (helped vs harmed)
        intervention_helped = False
        intervention_harmed = False

        for i, step in enumerate(steps):
            if step.policy_action in ("REVERT", "RESET") and i < len(steps) - 1:
                prev_correct = step.correctness
                next_correct = steps[i + 1].correctness

                if prev_correct == 0 and next_correct == 1:
                    intervention_helped = True
                elif prev_correct == 1 and next_correct == 0:
                    intervention_harmed = True

        # Energy metrics
        energies = [s.total_energy for s in steps]
        avg_energy = sum(energies) / len(energies)
        max_energy = max(energies)

        return PolicyEvaluationDTO(
            run_id=run_id,
            problem_id=problem_id,
            policy_id=policy_id,
            seed=seed,
            experiment_id=experiment_id,
            final_correct=final_correct,
            any_intervention=any_intervention,
            intervention_helped=intervention_helped,
            intervention_harmed=intervention_harmed,
            num_iterations=len(steps),
            avg_energy=avg_energy,
            max_energy=max_energy,
            tau_soft=thresholds["tau_soft"],
            tau_medium=thresholds["tau_medium"],
            tau_hard=thresholds["tau_hard"],
            created_at=created_at,
        )

    def generate_report(self, run_id: str) -> str:
        """Generate a human-readable Stage 2 report for a single run."""
        import pandas as pd

        evaluations = self.memory.policy_evaluations.get_by_run_id(run_id)
        if not evaluations:
            return "No Stage 2 evaluations found for this run."

        df = pd.DataFrame([e.model_dump() for e in evaluations])

        # ==========================================================
        # CURRENT RUN METRICS
        # ==========================================================
        accuracy = df["final_correct"].mean() * 100
        intervention_rate = df["any_intervention"].mean() * 100
        helped = int(df["intervention_helped"].sum())
        harmed = int(df["intervention_harmed"].sum())
        avg_energy = df["avg_energy"].mean()
        avg_iterations = df["num_iterations"].mean()

        # Thresholds: use first row (all problems in run share same thresholds)
        tau_soft = df["tau_soft"].iloc[0] if len(df) > 0 else 0.0
        tau_medium = df["tau_medium"].iloc[0] if len(df) > 0 else 0.0
        tau_hard = df["tau_hard"].iloc[0] if len(df) > 0 else 0.0

        # ==========================================================
        # HISTORICAL COMPARISON (same policy_id, exclude current run)
        # ==========================================================
        current_policy_id = int(df["policy_id"].iloc[0]) if len(df) > 0 else None
        historical_comparison = None
        
        if current_policy_id is not None:
            # Fetch historical evaluations for same policy, excluding current run
            historical = self.memory.policy_evaluations.get_by_policy_id(
                policy_id=current_policy_id,
                experiment_id=None  # cross-experiment comparison
            )
            # Filter out current run
            historical = [e for e in historical if e.run_id != run_id]
            
            if historical:
                hist_df = pd.DataFrame([e.model_dump() for e in historical])
                historical_comparison = {
                    "n_runs": hist_df["run_id"].nunique(),
                    "n_problems": len(hist_df),
                    "avg_accuracy": hist_df["final_correct"].mean() * 100,
                    "avg_intervention_rate": hist_df["any_intervention"].mean() * 100,
                    "total_helped": int(hist_df["intervention_helped"].sum()),
                    "total_harmed": int(hist_df["intervention_harmed"].sum()),
                    "avg_energy": hist_df["avg_energy"].mean(),
                }

        # ==========================================================
        # BUILD REPORT
        # ==========================================================
        
        # Helper for delta formatting
        def fmt_delta(current: float, historical: float) -> str:
            delta = current - historical
            sign = "+" if delta >= 0 else ""
            return f"{sign}{delta:.2f}%"

        report_lines = [
            "╔══════════════════════════════════════════════════════════════╗",
            "║              STAGE 2 EVALUATION REPORT                       ║",
            "╠══════════════════════════════════════════════════════════════╣",
            f"║ Run ID: {run_id:<50} ║",
            f"║ Policy ID: {current_policy_id:<45} ║",
            "╠══════════════════════════════════════════════════════════════╣",
            "║ CURRENT RUN METRICS                                          ║",
            "╠──────────────────────────────────────────────────────────────╣",
            f"║ Accuracy:              {accuracy:>6.2f}%                     ║",
            f"║ Intervention Rate:     {intervention_rate:>6.2f}%            ║",
            f"║ Interventions Helped:  {helped:>6d}                          ║",
            f"║ Interventions Harmed:  {harmed:>6d}                          ║",
            f"║ Avg Energy:            {avg_energy:>6.4f}                    ║",
            f"║ Avg Iterations:        {avg_iterations:>6.2f}                ║",
            "╠──────────────────────────────────────────────────────────────╣",
            "║ THRESHOLDS USED                                              ║",
            "╠──────────────────────────────────────────────────────────────╣",
            f"║ Tau Soft:   {tau_soft:>6.4f}                                 ║",
            f"║ Tau Medium: {tau_medium:>6.4f}                               ║",
            f"║ Tau Hard:   {tau_hard:>6.4f}                                 ║",
        ]

        # Add historical comparison section if data exists
        if historical_comparison:
            hist = historical_comparison
            acc_delta = fmt_delta(accuracy, hist["avg_accuracy"])
            int_delta = fmt_delta(intervention_rate, hist["avg_intervention_rate"])
            
            # Determine overall verdict
            if accuracy > hist["avg_accuracy"] and harmed <= hist["total_harmed"] / max(1, hist["n_runs"]):
                verdict = "✅ OUTPERFORMS HISTORY"
            elif accuracy < hist["avg_accuracy"] and harmed > hist["total_harmed"] / max(1, hist["n_runs"]):
                verdict = "❌ UNDERPERFORMS HISTORY"
            else:
                verdict = "⚠️  COMPARABLE TO HISTORY"

            report_lines.extend([
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
            report_lines.extend([
                "╠══════════════════════════════════════════════════════════════╣",
                "║ COMPARISON TO PREVIOUS RUNS                                  ║",
                "╠──────────────────────────────────────────────────────────────╣",
                "║ No historical data available for this policy.                ║",
                "║ Run more evaluations to enable comparison.                   ║",
            ])

        report_lines.append("╚══════════════════════════════════════════════════════════════╝")

        report = "\n".join(report_lines)
        logger.info("Stage 2 report generated for run: %s", run_id)
        return report
