# constrain/experiments/recovery_experiment.py
from __future__ import annotations

import copy
import json
import logging
from typing import Any, Dict, List, Optional

import numpy as np

from constrain.config import get_config
from constrain.data.memory import Memory
from constrain.energy.embedding.hf_embedder import HFEmbedder
from constrain.energy.embedding.sqlite_embedding_backend import \
    SQLiteEmbeddingBackend
from constrain.energy.gate import VerifiabilityGate
from constrain.energy.geometry.claim_evidence import ClaimEvidenceGeometry
from constrain.core.model import call_model
from constrain.reasoning_state import ReasoningState

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class RecoveryExperiment:

    def __init__(self, memory: Memory, gate: VerifiabilityGate, tau_hard: float):
        self.memory = memory
        self.gate = gate
        self.tau_hard = tau_hard

    # ============================================================
    # PUBLIC ENTRY
    # ============================================================

    def run(self, run_id: str, max_depth: int = 20) -> List[Dict[str, Any]]:
        logger.info(f"Running recovery experiment on run_id={run_id} with tau_hard={self.tau_hard:.4f}")

        conversations = self._load_conversations(run_id)
        logger.info(f"Loaded {len(conversations)} conversations")

        results = []

        for idx, conv in enumerate(conversations):
            logger.info(f"[{idx+1}/{len(conversations)}] Processing problem_id={conv[0].problem_id}")

            result = self._run_single_conversation(conv, max_depth)

            if result:
                logger.info(
                    f"  Spike at iter {result['spike_iteration']} | "
                    f"Δ survival = {result.get('delta_survival', 0.0):.3f} | "
                )
                results.append(result)
            else:
                logger.info("  No spike detected.")

        return results

    # ============================================================
    # CONVERSATION LOADING
    # ============================================================

    def _load_conversations(self, run_id: str):
        steps = self.memory.steps.get_by_run(run_id)

        conversations = {}
        for step in steps:
            key = step.problem_id
            conversations.setdefault(key, []).append(step)

        # Sort by iteration
        for k in conversations:
            conversations[k] = sorted(
                conversations[k],
                key=lambda s: s.iteration
            )

        return list(conversations.values())

    # ============================================================
    # CORE LOGIC
    # ============================================================

    def _run_single_conversation(self, steps, max_depth):

        prompt = steps[0].prompt_text
        base_temp = steps[0].temperature or 0.7

        state = ReasoningState(prompt, temperature=base_temp)

        spike_iteration = None

        # --- Replay original trajectory until spike ---
        for step in steps:
            state.accept(step.reasoning_text)

            if step.total_energy > self.tau_hard:
                spike_iteration = step.iteration
                break

        if spike_iteration is None:
            return None

        # --- Clone state for three arms ---
        state_A = copy.deepcopy(state)
        state_B = copy.deepcopy(state)
        state_C = copy.deepcopy(state)

        # --- Apply interventions ---
        # A = baseline (do nothing)

        # B = temperature reduction
        state_B.temperature = max(0.2, state_B.temperature * 0.7)

        # C = revert + temperature reduction
        if len(state_C.history) >= 2:
            state_C.revert()
        state_C.temperature = max(0.2, state_C.temperature * 0.7)

        # --- Continue trajectories ---
        metrics_A = self._continue(state_A, max_depth)
        metrics_B = self._continue(state_B, max_depth)
        metrics_C = self._continue(state_C, max_depth)

        return {
            "problem_id": steps[0].problem_id,
            "spike_iteration": spike_iteration,
            "A": metrics_A,
            "B": metrics_B,
            "C": metrics_C,
        }

    # ============================================================
    # CONTINUE TRAJECTORY
    # ============================================================

    def _continue(self, state: ReasoningState, max_depth: int):

        survival = 0
        avoided_collapse = True
        min_margin = float("inf")

        for i in range(max_depth):

            prompt_text = f"Solve step by step:\n\n{state.current}"
            reasoning = call_model(prompt_text, state.temperature)

            energy, axes, _ = self.gate.compute_axes(
                claim=reasoning,
                evidence_texts=[state.prompt] + state.history
            )
            print(f"  Iter {i+1} | Energy: {energy.energy:.4f} | Tau hard: {self.tau_hard:.4f} \n {json.dumps(energy.to_dict(), indent=2)}")

            margin = self.tau_hard - energy.energy
            min_margin = min(min_margin, margin)

            if energy.energy > self.tau_hard:
                avoided_collapse = False
                break

            state.accept(reasoning)
            survival += 1

        return {
            "survival": survival,
            "avoided_collapse": avoided_collapse,
            "min_energy_margin": min_margin,
        }
    

# ============================================================
# RUN SELECTION LOGIC
# ============================================================

def find_recent_large_run(memory: Memory, min_steps: int = 100) -> Optional[str]:
    """
    Find the most recent completed run with >= min_steps.
    """

    runs = memory.runs.get_all(desc_start_time=True)

    for run in runs:
        if run.status != "completed":
            continue

        step_count = memory.steps.count_by_run(run.run_id)

        if step_count >= min_steps:
            logger.info(f"Selected run {run.run_id} with {step_count} steps")
            return run.run_id

    logger.warning("No suitable run found.")
    return None


# ============================================================
# MAIN
# ============================================================

def main():

    cfg = get_config()
    memory = Memory(cfg.db_url)

    # --------------------------------------------------------
    # Select run
    # --------------------------------------------------------

    run_id = find_run_with_spike(memory)
    run_id = "run_2898a03a"

    # run_id = find_recent_large_run(memory, min_steps=100)

    if run_id is None:
        logger.error("❌ No run found for recovery experiment.")
        return

    # --------------------------------------------------------
    # Rebuild gate (same as runner)
    # --------------------------------------------------------

    embedder = HFEmbedder(
        model_name=cfg.embedding_model,
        backend=SQLiteEmbeddingBackend(str(cfg.embedding_db)),
    )

    energy_computer = ClaimEvidenceGeometry(top_k=6, rank_r=4)

    gate = VerifiabilityGate(
        embedder=embedder,
        energy_computer=energy_computer,
    )

    # --------------------------------------------------------
    # Use calibrated tau_hard if available
    # --------------------------------------------------------

    run_obj = memory.runs.get_by_id(run_id)
    tau_hard = run_obj.tau_hard_calibrated or run_obj.tau_hard

    logger.info(f"Using tau_hard={tau_hard:.4f}")

    # --------------------------------------------------------
    # Run Recovery Experiment
    # --------------------------------------------------------

    experiment = RecoveryExperiment(
        memory=memory,
        gate=gate,
        tau_hard=tau_hard,
    )

    results = experiment.run(run_id=run_id, max_depth=20)

    if not results:
        logger.warning("No energy spikes found in this run.")
        return

    # --------------------------------------------------------
    # Aggregate Results
    # --------------------------------------------------------

    deltas = [r["delta_survival"] for r in results]

    mean_delta = np.mean(deltas)
    std_delta = np.std(deltas)

    improved = sum(1 for d in deltas if d > 0)
    harmed = sum(1 for d in deltas if d < 0)

    logger.info("=" * 60)
    logger.info("RECOVERY EXPERIMENT RESULTS")
    logger.info("=" * 60)
    logger.info(f"Run ID:              {run_id}")
    logger.info(f"Conversations tested:{len(results)}")
    logger.info(f"Mean Δ Survival:     {mean_delta:.3f}")
    logger.info(f"Std Δ Survival:      {std_delta:.3f}")
    logger.info(f"Improved cases:      {improved}/{len(results)}")
    logger.info(f"Harmed cases:        {harmed}/{len(results)}")
    logger.info("=" * 60)

    if mean_delta > 0:
        logger.info("✅ Intervention appears to extend survival.")
    elif mean_delta < 0:
        logger.info("⚠️ Intervention appears harmful.")
    else:
        logger.info("⚠️ No measurable effect detected.")


def find_run_with_spike(memory):
    runs = memory.runs.get_all(desc_start_time=True)

    for run in runs:
        steps = memory.steps.get_by_run(run.run_id)
        if not steps:
            continue

        energies = [s.total_energy for s in steps if s.total_energy is not None]
        if not energies:
            continue

        max_energy = max(energies)
        tau = run.tau_hard_calibrated or run.tau_hard

        if max_energy > tau:
            print("=" * 60)
            print("Found candidate run:", run.run_id)
            print("Max energy:", max_energy)
            print("Tau hard:", tau)
            print("=" * 60)
            return run.run_id

    print("No runs exceed tau_hard.")
    return None

if __name__ == "__main__":
    main()