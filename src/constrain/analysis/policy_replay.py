from __future__ import annotations

import numpy as np
from typing import Dict, Any, List
from collections import defaultdict

from constrain.data.memory import Memory
from constrain.policy import apply_policy
from constrain.config import get_config


class PolicyReplayService:

    @staticmethod
    def replay_run(
        memory: Memory,
        run_id: str,
        policy_id: int,
        tau_soft: float,
        tau_medium: float,
        tau_hard: float,
    ) -> Dict[str, Any]:

        # -------------------------------------------------
        # Load steps ordered properly
        # -------------------------------------------------

        steps = memory.steps.list(
            filters={"run_id": run_id},
            order_by="iteration",
            desc=False,
            limit=100000,
        )

        if not steps:
            raise ValueError(f"No steps found for run_id={run_id}")

        # -------------------------------------------------
        # Override config thresholds temporarily
        # -------------------------------------------------

        stats = defaultdict(int)
        energy_values = []

        temperature = None
        last_stable = None

        for step in steps:

            energy = step.total_energy
            energy_values.append(energy)

            # Initialize
            if temperature is None:
                temperature = step.temperature
            if last_stable is None:
                last_stable = step.reasoning_text

            # Simulate policy (NO model call)
            new_state, temperature, action = PolicyReplayService._simulate_policy(
                policy_id,
                energy,
                last_stable,
                step.prompt_text,
                temperature,
                tau_soft,
                tau_medium,
                tau_hard,
            )

            stats[action] += 1

            if action == "ACCEPT":
                last_stable = step.reasoning_text

        total = len(steps)

        return {
            "run_id": run_id,
            "policy_id": policy_id,
            "tau_soft": tau_soft,
            "tau_medium": tau_medium,
            "tau_hard": tau_hard,
            "total_steps": total,
            "accept_rate": stats["ACCEPT"] / total,
            "intervention_rate": 1.0 - (stats["ACCEPT"] / total),
            "action_counts": dict(stats),
            "energy_mean": float(np.mean(energy_values)),
            "energy_std": float(np.std(energy_values)),
        }

    # -------------------------------------------------
    # Internal simulation logic (no config reads)
    # -------------------------------------------------

    @staticmethod
    def _simulate_policy(
        policy_id: int,
        energy: float,
        last_stable: str,
        prompt: str,
        temperature: float,
        tau_soft: float,
        tau_medium: float,
        tau_hard: float,
    ):

        if policy_id == 0:
            return last_stable, temperature, "ACCEPT"

        if energy <= tau_soft:
            return last_stable, temperature, "ACCEPT"

        if policy_id == 1:
            return last_stable, temperature, "REVERT"

        if policy_id == 2:
            return last_stable, temperature * 0.9, "REVERT_COOL"

        if policy_id == 3:
            if energy > tau_medium:
                return last_stable, temperature * 0.75, "REVERT_AGGRESSIVE"
            return last_stable, temperature, "REVERT"

        if policy_id == 4:
            if energy > tau_hard:
                return prompt, temperature * 0.7, "RESET_PROMPT"
            return last_stable, temperature, "REVERT"

        if policy_id == 5:
            if energy > tau_medium:
                return prompt, temperature * 0.85, "RESET"
            return last_stable, temperature * 0.85, "REVERT_STABILIZE"

        return last_stable, temperature, "ACCEPT"
