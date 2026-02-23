# constrain/runner.py
from __future__ import annotations

import logging
import random
import time
import uuid
from typing import Optional, Type

import numpy as np
from datasets import load_dataset
from tqdm.auto import tqdm

from constrain.analysis.aggregation.metrics_aggregator import MetricsAggregator
from constrain.analysis.aggregation.metrics_calculator import MetricsCalculator
from constrain.analysis.aggregation.populate_problem_summaries import populate_for_run
from constrain.config import get_config
from constrain.data.memory import Memory
from constrain.data.schemas.intervention import InterventionDTO
from constrain.data.schemas.run import RunDTO
from constrain.data.schemas.step import StepDTO
from constrain.energy.embedding.hf_embedder import HFEmbedder
from constrain.energy.embedding.sqlite_embedding_backend import SQLiteEmbeddingBackend
from constrain.energy.gate import VerifiabilityGate
from constrain.energy.geometry.claim_evidence import ClaimEvidenceGeometry
from constrain.energy.utils.text_utils import split_into_sentences
from constrain.model import call_model
from constrain.policy.custom_types import PolicyDecision, ThresholdProvider
from constrain.policy.engine import PolicyEngine
from constrain.policy.registry import PolicyRegistry
from constrain.policy.thresholds import CalibrationThresholdProvider
from constrain.reasoning_state import ReasoningState
from constrain.utils.dict_utils import flatten_numeric_dict

logger = logging.getLogger(__name__)


# =============================================================================
# PUBLIC API — Thin Wrappers
# =============================================================================

def run(
    policy_id: int = 4,
    seed: int = 42,
    num_problems: Optional[int] = None,
    threshold: Optional[float] = None,
) -> str:
    """
    Run experiment with default CalibrationThresholdProvider.
    
    This is the standard entry point for most experiments.
    """
    threshold_provider = CalibrationThresholdProvider()
    return run_experiment(
        policy_id=policy_id,
        seed=seed,
        num_problems=num_problems,
        threshold_provider=threshold_provider,
        threshold_override=threshold,
    )


def run_with_provider(
    policy_id: int,
    threshold_provider: ThresholdProvider,
    seed: int = 42,
    num_problems: Optional[int] = None,
) -> str:
    """
    Run experiment with injected ThresholdProvider.
    
    Use this for threshold sweeps, calibration experiments, or custom policies.
    """
    return run_experiment(
        policy_id=policy_id,
        seed=seed,
        num_problems=num_problems,
        threshold_provider=threshold_provider,
        threshold_override=None,
    )


# =============================================================================
# CORE EXPERIMENT RUNNER — Single Source of Truth
# =============================================================================

def run_experiment(
    *,
    policy_id: int,
    seed: int,
    num_problems: Optional[int],
    threshold_provider: ThresholdProvider,
    threshold_override: Optional[float] = None,
) -> str:
    """
    Core experiment runner — all logic lives here.
    
    Args:
        policy_id: Which policy to use from PolicyRegistry
        seed: Random seed for reproducibility
        num_problems: Number of problems to evaluate (uses config default if None)
        threshold_provider: Strategy for computing tau_soft/medium/hard
        threshold_override: Optional single threshold value (for legacy support)
    
    Returns:
        run_id: Unique identifier for this experiment run
    """
    start_time = time.time()
    logger.info("🚀 Starting experiment: policy=%d, seed=%d", policy_id, seed)

    # ─────────────────────────────────────────────────────────────
    # STAGE 0: Initialization
    # ─────────────────────────────────────────────────────────────
    random.seed(seed)
    np.random.seed(seed)
    cfg = get_config()

    memory = Memory(cfg.db_url)
    run_id = f"run_{uuid.uuid4().hex[:8]}"

    # Initialize components
    embedder = HFEmbedder(
        model_name=cfg.embedding_model,
        backend=SQLiteEmbeddingBackend(str(cfg.embedding_db)),
    )
    energy_computer = ClaimEvidenceGeometry(top_k=6, rank_r=4)
    gate = VerifiabilityGate(embedder=embedder, energy_computer=energy_computer)

    # Register run in database
    run_dto = _create_run_dto(
        run_id=run_id,
        cfg=cfg,
        policy_id=policy_id,
        seed=seed,
        num_problems=num_problems,
        threshold_provider=threshold_provider,
        threshold_override=threshold_override,
        start_time=start_time,
    )
    memory.runs.create(run_dto)

    thresholds = threshold_provider.get(
        cfg=cfg,
        memory=memory,
        run_id=run_id,
    )

    memory.calibrations.create_calibration(
        run_id=run_id,
        policy_mode="dynamic",
        tau_soft=thresholds.tau_soft,
        tau_medium=thresholds.tau_medium,
        tau_hard=thresholds.tau_hard,
        sample_count=None,
    )

    # Initialize policy engine
    engine = PolicyEngine(
        policy=PolicyRegistry.from_id(policy_id),
        threshold_provider=threshold_provider,
    )

    # ─────────────────────────────────────────────────────────────
    # STAGE 1: Load Dataset
    # ─────────────────────────────────────────────────────────────
    dataset = load_dataset("gsm8k", "main", split="test")
    dataset = dataset.shuffle(seed=seed).select(range(num_problems or cfg.num_problems))
    logger.info("📊 Loaded %d problems for evaluation", len(dataset))

    # ─────────────────────────────────────────────────────────────
    # STAGE 2: Main Problem Loop (extracted for clarity)
    # ─────────────────────────────────────────────────────────────
    _run_problem_loop(
        dataset=dataset,
        cfg=cfg,
        memory=memory,
        gate=gate,
        engine=engine,
        run_id=run_id,
        seed=seed,
    )

    # ─────────────────────────────────────────────────────────────
    # STAGE 3: Post-Processing
    # ─────────────────────────────────────────────────────────────
    _finalize_run(
        memory=memory,
        run_id=run_id,
        start_time=start_time,
        cfg=cfg,
    )

    logger.info("✅ Experiment %s completed in %.2fs", run_id, time.time() - start_time)
    return run_id


# =============================================================================
# HELPER: Create RunDTO with proper threshold handling
# =============================================================================

def _create_run_dto(
    *,
    run_id: str,
    cfg,
    policy_id: int,
    seed: int,
    num_problems: Optional[int],
    threshold_provider: ThresholdProvider,
    threshold_override: Optional[float],
    start_time: float,
) -> RunDTO:
    """
    Create RunDTO with thresholds from provider (or override).
    
    This isolates the threshold logic so it's easy to test and modify.
    """
    # Get thresholds from provider
    thresholds = threshold_provider.get(cfg=cfg, memory=Memory(cfg.db_url), run_id=run_id)
    
    # Allow override for legacy single-threshold experiments
    if threshold_override is not None:
        tau_soft = tau_medium = tau_hard = threshold_override
    else:
        tau_soft = thresholds.tau_soft
        tau_medium = thresholds.tau_medium
        tau_hard = thresholds.tau_hard

    return RunDTO(
        run_id=run_id,
        model_name=cfg.model_name,
        initial_temperature=cfg.initial_temperature,
        num_problems=num_problems or cfg.num_problems,
        num_recursions=cfg.num_recursions,
        tau_soft=tau_soft,
        tau_medium=tau_medium,
        tau_hard=tau_hard,
        policy_id=policy_id,
        task_type="gsm8k",
        start_time=start_time,
        status="running",
        notes=cfg.notes,
        seed=seed,
    )


# =============================================================================
# HELPER: Main problem loop (extracted for readability)
# =============================================================================

def _run_problem_loop(
    *,
    dataset,
    cfg,
    memory: Memory,
    gate: VerifiabilityGate,
    engine: PolicyEngine,
    run_id: str,
    seed: int,
):
    """
    Execute the main evaluation loop over problems.
    
    This is the core inference + policy + logging logic.
    """
    for pid, example in enumerate(tqdm(dataset, desc="Problems")):
        _run_single_problem(
            pid=pid,
            example=example,
            cfg=cfg,
            memory=memory,
            gate=gate,
            engine=engine,
            run_id=run_id,
            seed=seed,
        )


def _run_single_problem(
    *,
    pid: int,
    example: dict,
    cfg,
    memory: Memory,
    gate: VerifiabilityGate,
    engine: PolicyEngine,
    run_id: str,
    seed: int,
):
    """
    Run a single problem through all iterations.
    
    Handles: model calls, energy computation, policy decisions, state updates.
    """
    prompt = example["question"]
    gold_answer = example["answer"].split("####")[-1].strip()
    
    state = ReasoningState(prompt)
    state.temperature = cfg.initial_temperature

    for iteration in range(cfg.num_recursions):
        try:
            # ─────────────────────────────────────────────────────
            # 1. Model Generation (with caching)
            # ─────────────────────────────────────────────────────
            prompt_text = f"Solve step by step:\n\n{state.current}"
            
            cached = memory.steps.get_reasoning_by_prompt(prompt, state.temperature)
            if cached:
                reasoning = cached.reasoning_text
            else:
                reasoning = call_model(prompt_text, state.temperature)

            # ─────────────────────────────────────────────────────
            # 2. Evidence + Energy Computation
            # ─────────────────────────────────────────────────────
            evidence_texts = _build_evidence(prompt, state.history)
            energy_result, axes, _ = gate.compute_axes(
                claim=reasoning,
                evidence_texts=evidence_texts,
            )
            
            stability_energy = _compute_stability_energy(reasoning, state.history, gate)
            total_energy = axes.get("energy")

            # ─────────────────────────────────────────────────────
            # 3. Metrics Computation
            # ─────────────────────────────────────────────────────
            all_metrics = MetricsCalculator.compute_all(
                reasoning=reasoning,
                gold_answer=gold_answer,
                energy_metrics=energy_result.to_dict(),
                cfg=cfg,
            )
            
            # Merge energy axes into metrics
            all_metrics.update({
                "energy": axes.get("energy"),
                "participation_ratio": axes.get("participation_ratio"),
                "sensitivity": axes.get("sensitivity"),
                "alignment": axes.get("alignment"),
                "sim_margin": axes.get("sim_margin"),
                "iteration": iteration,
                "evidence_count": len(state.history),
            })
            flat_metrics = flatten_numeric_dict(all_metrics)

            # ─────────────────────────────────────────────────────
            # 4. Policy Decision + State Update
            # ─────────────────────────────────────────────────────
            decision: PolicyDecision = engine.apply(
                axes=axes,
                flat_metrics=flat_metrics,
                state=state,
                memory=memory,
                run_id=run_id,
                step_id=None,
            )

            _apply_policy_action(state, decision.action, decision.new_temperature)

            # ─────────────────────────────────────────────────────
            # 5. Persistence: Step + Metrics + Interventions
            # ─────────────────────────────────────────────────────
            step_dto = _create_step_dto(
                run_id=run_id,
                problem_id=pid,
                iteration=iteration,
                prompt_text=prompt,
                reasoning_text=reasoning,
                gold_answer=gold_answer,
                all_metrics=all_metrics,
                axes=axes,
                stability_energy=stability_energy,
                decision=decision,
                state=state,
            )
            step_dto = memory.steps.create(step_dto)

            engine.log_policy_event(
                memory=memory,
                run_id=run_id,
                step_id=step_dto.id,
                decision=decision,
            )

            memory.metrics.bulk_from_dict(
                step_id=step_dto.id,
                stage="energy_v2",
                metrics=flat_metrics,
            )

            if decision.action != "ACCEPT":
                _log_intervention(
                    memory=memory,
                    run_id=run_id,
                    problem_id=pid,
                    iteration=iteration,
                    action=decision.action,
                    new_temperature=decision.new_temperature,
                )

        except Exception as e:
            logger.exception("❌ Crash at problem %d, iteration %d: %s", pid, iteration, e)
            break  # Stop this problem, continue to next


# =============================================================================
# HELPER: Utility functions (small, focused, testable)
# =============================================================================

def _build_evidence(prompt: str, history: list[str]) -> list[str]:
    """Build evidence list from prompt + history."""
    evidence = split_into_sentences(prompt)
    for past in history:
        evidence.extend(split_into_sentences(past))
    return evidence


def _compute_stability_energy(
    claim: str,
    history: list[str],
    gate: VerifiabilityGate,
) -> float:
    """Compute stability energy vs last accepted step."""
    if not history:
        return 0.0
    
    last = history[-1]
    _, stability_axes, _ = gate.compute_axes(
        claim=claim,
        evidence_texts=split_into_sentences(last),
    )
    return stability_axes.get("energy", 0.0)


def _apply_policy_action(state: ReasoningState, action: str, new_temperature: float):
    """Apply policy action to reasoning state."""
    if action == "ACCEPT":
        state.accept(state.current or "")
    elif action == "REVERT":
        state.revert()
    elif action in ("RESET", "RESET_PROMPT"):
        state.reset()
    
    state.temperature = new_temperature


def _create_step_dto(
    *,
    run_id: str,
    problem_id: int,
    iteration: int,
    prompt_text: str,
    reasoning_text: str,
    gold_answer: str,
    all_metrics: dict,
    axes: dict,
    stability_energy: float,
    decision: PolicyDecision,
    state: ReasoningState,
) -> StepDTO:
    """Create StepDTO from all computed values."""
    return StepDTO(
        run_id=run_id,
        problem_id=problem_id,
        iteration=iteration,
        prompt_text=prompt_text,
        reasoning_text=reasoning_text,
        collapse_probability=decision.collapse_probability,
        gold_answer=gold_answer,
        extracted_answer=all_metrics["extracted_answer"],
        total_energy=axes.get("energy"),
        grounding_energy=axes.get("energy"),  # or energy_result.energy if separate
        stability_energy=stability_energy,
        correctness=all_metrics.get("correctness"),
        accuracy=all_metrics.get("accuracy"),
        temperature=state.temperature,
        policy_action=decision.action,
        phase=MetricsCalculator.PHASE_VALUE_TO_LABEL[all_metrics["phase_value"]],
        timestamp=time.time(),
    )


def _log_intervention(
    *,
    memory: Memory,
    run_id: str,
    problem_id: int,
    iteration: int,
    action: str,
    new_temperature: float,
):
    """Log intervention event for audit trail."""
    intervention = InterventionDTO(
        run_id=run_id,
        problem_id=problem_id,
        iteration=iteration,
        threshold="learned",
        rationale=action,
        reverted_to=iteration - 1,
        new_temperature=new_temperature,
        timestamp=time.time(),
    )
    memory.interventions.create(intervention)


def _finalize_run(
    *,
    memory: Memory,
    run_id: str,
    start_time: float,
    cfg,
):
    """
    Post-run tasks: aggregation, evaluation, signal discovery, cleanup.
    """
    # 1. Metrics aggregation
    try:
        MetricsAggregator.dump_run_csv(memory, run_id)
    except Exception as e:
        logger.exception("⚠️  Aggregation failed: %s", e)

    # 2. Mark run complete
    memory.runs.update(run_id, {"status": "completed", "end_time": time.time()})

    # 3. Stage 2 evaluation
    try:
        from constrain.services.policy_evaluation_service import PolicyEvaluationService
        evaluator = PolicyEvaluationService(memory)
        evaluations = evaluator.evaluate_run(run_id)
        report = evaluator.generate_report(run_id)
        logger.info("📊 Stage 2 evaluation: %s", report)
    except Exception as e:
        logger.exception("⚠️  Stage 2 evaluation failed: %s", e)

    # 4. Signal discovery (optional)
    if cfg.run_signal_discovery:
        try:
            from constrain.services.collapse_prediction_service import CollapsePredictionService
            signal_service = CollapsePredictionService(memory)
            results = signal_service.discover_signals(
                run_id=run_id,
                prediction_horizon=cfg.signal_prediction_horizon,
                save_plots=cfg.save_signal_plots,
            )
            if not results.get("skipped"):
                signal_service.persist_signal_report(
                    run_id=run_id,
                    results=results,
                    experiment_id=None,  # Add if using experiments table
                )
        except Exception as e:
            logger.exception("⚠️  Signal discovery failed: %s", e)

    # 5. Populate problem_summaries (critical for utility head training)
    try:
        populate_for_run(memory, run_id)
    except Exception as e:
        logger.exception("⚠️  problem_summaries population failed: %s", e)