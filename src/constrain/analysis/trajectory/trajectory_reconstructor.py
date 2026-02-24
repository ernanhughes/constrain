from __future__ import annotations

import logging
from typing import Dict, List, Optional

import pandas as pd

from constrain.config import get_config
from constrain.data.memory import Memory
from constrain.data.schemas.step import StepDTO

logger = logging.getLogger(__name__)


class TrajectoryReconstructor:
    """
    Reconstructs problem-level trajectories from raw step logs.
    
    This is a shared utility used by:
    - PolicyEvaluationService (policy outcomes)
    - CollapsePredictionService (signal discovery)
    - Any future analysis needing problem-level aggregation
    
    Operates strictly on persisted logs — no recomputation.
    """

    def __init__(self, memory: Memory):
        self.memory = memory

    def build_problem_dataframe(self, run_id: str) -> pd.DataFrame:
        """
        Reconstruct problem-level outcomes from step trajectories.
        
        Returns:
            DataFrame with one row per problem, columns:
            - problem_id, final_correct, any_intervention, num_interventions,
              avg_energy, max_energy, num_iterations, intervention_helped,
              intervention_harmed, collapsed
        """
        # Load steps for this run
        steps = self.memory.steps.get_by_run(run_id)
        if not steps:
            logger.warning(f"No steps found for run: {run_id}")
            return pd.DataFrame()

        # Load run metadata for thresholds
        run = self.memory.runs.get_by_id(run_id)
        if not run:
            raise ValueError(f"Run not found: {run_id}")

        # Group steps by problem
        problems: Dict[int, List[StepDTO]] = {}
        for step in steps:
            if step.problem_id not in problems:
                problems[step.problem_id] = []
            problems[step.problem_id].append(step)

        # Build rows
        rows = []
        cfg = get_config()

        for problem_id, problem_steps in problems.items():
            # Sort by iteration to ensure trajectory order
            problem_steps.sort(key=lambda s: s.iteration)

            if not problem_steps:
                continue

            # Final correctness (last iteration)
            final_correct = bool(problem_steps[-1].correctness) if problem_steps[-1].correctness is not None else False

            # Intervention flags
            any_intervention = any(s.policy_action in ("REVERT", "RESET") for s in problem_steps)
            num_interventions = sum(1 for s in problem_steps if s.policy_action in ("REVERT", "RESET"))

            # Energy metrics
            energies = [s.total_energy for s in problem_steps]
            avg_energy = sum(energies) / len(energies) if energies else 0.0
            max_energy = max(energies) if energies else 0.0

            # Collapse detection (energy > tau_hard at any point)
            collapsed = any(e > cfg.tau_hard for e in energies)

            # Intervention effectiveness: did correctness change after intervention?
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

            rows.append({
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
                # Include raw fields for flexibility
                "run_id": run_id,
                "policy_id": run.policy_id,
                "seed": run.seed,
                "experiment_id": run.experiment_id,
            })

        return pd.DataFrame(rows)

    def get_step_dataframe(self, run_id: str) -> pd.DataFrame:
        """
        Return raw steps as DataFrame (for services needing step-level access).
        
        Returns:
            DataFrame with one row per step.
        """
        steps = self.memory.steps.get_by_run(run_id)
        if not steps:
            return pd.DataFrame()

        rows = [s.model_dump() for s in steps]
        return pd.DataFrame(rows)

    def get_policy_events_dataframe(self, run_id: str) -> pd.DataFrame:
        """
        Return policy events as DataFrame (for threshold/decision analysis).
        
        Returns:
            DataFrame with one row per policy event.
        """
        events = self.memory.policy_events.get_by_run_id(run_id)
        if not events:
            return pd.DataFrame()

        rows = [e.model_dump() for e in events]
        return pd.DataFrame(rows)