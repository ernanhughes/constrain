# constrain/runner.py
from __future__ import annotations

import logging
import random
import time
import uuid
from typing import List, Optional

import numpy as np
from datasets import load_dataset
from tqdm.auto import tqdm

from constrain.evaluation.metrics.metrics_aggregator import MetricsAggregator
from constrain.evaluation.metrics.metrics_calculator import MetricsCalculator
from constrain.evaluation.metrics.populate_problem_summaries import \
    populate_for_run
from constrain.config import get_config
from constrain.data.memory import Memory
from constrain.data.schemas.intervention import InterventionDTO
from constrain.data.schemas.run import RunDTO
from constrain.energy.embedding.hf_embedder import HFEmbedder
from constrain.energy.embedding.sqlite_embedding_backend import \
    SQLiteEmbeddingBackend
from constrain.energy.gate import VerifiabilityGate
from constrain.energy.geometry.claim_evidence import ClaimEvidenceGeometry
from constrain.energy.utils.text_utils import split_into_sentences
from constrain.core.model import call_model
from constrain.policy.custom_types import ThresholdProvider
from constrain.policy.threshold.thresholds import CalibrationThresholdProvider
from constrain.reasoning_state import ReasoningState
from constrain.utils.dict_utils import flatten_numeric_dict

from constrain.control import (
    classify_violation,
    compute_slope,
    update_counters,
    is_collapse,
    reset_dynamics,
    EnergyControlPolicy,
    EnergyDynamics,
    Action,
)

logger = logging.getLogger(__name__)


# =============================================================================
# PUBLIC API — Thin Wrappers
# =============================================================================

def run(
    policy_id: int = 4,
    seed: int = 42,
    num_problems: Optional[int] = None,
    num_recursions: Optional[int] = None,
    threshold: Optional[float] = None,
    initial_temperature: Optional[float] = None,
    cached_only: bool = False,  
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
        num_recursions=num_recursions,
        threshold_provider=threshold_provider,
        threshold_override=threshold,
        initial_temperature_override=initial_temperature,
        cached_only=cached_only,  # ← PASS IT
    )

def run_with_provider(
    policy_id: int,
    threshold_provider: ThresholdProvider,
    seed: int = 42,
    num_problems: Optional[int] = None,
    cached_only: bool = False, 
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
        cached_only=cached_only,
    )


# =============================================================================
# CORE EXPERIMENT RUNNER — Single Source of Truth
# =============================================================================

def run_experiment(
    *,
    policy_id: int,
    seed: int,
    num_problems: Optional[int],
    num_recursions: Optional[int],
    threshold_provider: ThresholdProvider,
    threshold_override: Optional[float] = None,
    initial_temperature_override: Optional[float] = None,
    cached_only: bool = False,  # ← NEW PARAM
) -> str:
    """
    Core experiment runner — all logic lives here.
    
    Args:
        policy_id: Which policy to use from PolicyRegistry
        seed: Random seed for reproducibility
        num_problems: Number of problems to evaluate (uses config default if None)
        threshold_provider: Strategy for computing tau_soft/medium/hard
        threshold_override: Optional single threshold value (for legacy support)
        cached_only: If True, replay from cache without generating new steps
    
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
        num_recursions=num_recursions,
        threshold_provider=threshold_provider,
        threshold_override=threshold_override,
        start_time=start_time,
        initial_temperature_override=initial_temperature_override,
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


    # ─────────────────────────────────────────────────────────────
    # STAGE 1: Load Dataset
    # ─────────────────────────────────────────────────────────────
    if cached_only:
        dataset = load_cached_problems(
            memory=memory,
            seed=seed,
            num_problems=num_problems or cfg.num_problems,
            min_steps=num_recursions or cfg.num_recursions,
        )
    else:
        dataset = load_dataset("gsm8k", "main", split="test")
        dataset = dataset.shuffle(seed=seed).select(range(num_problems or cfg.num_problems))

    logger.info("📊 Loaded %d problems for evaluation", len(dataset))



    # ─────────────────────────────────────────────────────────────
    # STAGE 1: Initialize policy engine
    # ─────────────────────────────────────────────────────────────
    policy = EnergyControlPolicy(
        min_temperature=cfg.min_temperature,
        cooldown_medium=cfg.revert_cooldown_factor,
        cooldown_hard_small_slope=cfg.aggressive_cooldown_factor,
        cooldown_hard_large_slope=cfg.reset_cooldown_factor,
        runaway_slope_eps=0.05,
    )

    # ─────────────────────────────────────────────────────────────
    # STAGE 3: Main Problem Loop (extracted for clarity)
    # ─────────────────────────────────────────────────────────────
    _run_problem_loop(
        dataset=dataset,
        cfg=cfg,
        memory=memory,
        gate=gate,
        policy=policy,  # ← Use control policy
        run_id=run_id,
        num_recursions=num_recursions,
        seed=seed,
        cached_only=cached_only,  # ← PASS IT
    )

    # ─────────────────────────────────────────────────────────────
    # STAGE 4: Post-Processing
    # ─────────────────────────────────────────────────────────────
    _finalize_run(
        memory=memory,
        run_id=run_id,
        start_time=start_time,
        cfg=cfg,
        cached_only=cached_only,  # ← PASS IT
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
    num_recursions: Optional[int], 
    threshold_provider: ThresholdProvider,
    threshold_override: Optional[float],
    start_time: float,
    initial_temperature_override: Optional[float],
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
        initial_temperature=(
            initial_temperature_override
            if initial_temperature_override is not None
            else cfg.initial_temperature
        ),
        num_problems=num_problems or cfg.num_problems,
        num_recursions=num_recursions or cfg.num_recursions,
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
    policy: EnergyControlPolicy,  # ← Control policy
    run_id: str,
    num_recursions: Optional[int],
    seed: int,
    cached_only: bool = False,  # ← NEW
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
            policy=policy,
            run_id=run_id,
            seed=seed,
            num_recursions=num_recursions,
            cached_only=cached_only,  # ← PASS IT
        )

def _run_single_problem(
    *,
    pid: int,
    example: dict,
    cfg,
    memory: Memory,
    gate: VerifiabilityGate,
    policy: EnergyControlPolicy,
    run_id: str,
    num_recursions: Optional[int],
    seed: int,
    max_attempts: int = 100,
    cached_only: bool = False,  
):
    """
    Run a single problem with control system.
    
    Clean state flow:
    1. Generate reasoning
    2. Compute energy
    3. Compute dynamics (violation, slope, counters)
    4. Check collapse (global)
    5. Policy decides
    6. Apply decision (push/pop/reset)
    7. Persist
    
    Key distinctions:
    - attempt: increments every model call (every loop)
    - iteration: increments only when we commit (ALLOW)
    - collapse: persistent hard + rising energy (containment failure)
    """
    
    prompt = example["question"]
    gold_answer = example["answer"].split("####")[-1].strip()
    
    run_obj = memory.runs.get_by_id(run_id)
    state = ReasoningState(
        prompt,
        temperature=run_obj.initial_temperature,
        run_id=run_id,
        problem_id=pid,
        snapshot_store=memory.reasoning_state_snapshots,
    )
    
    # ─────────────────────────────────────────────────────────────
    # DYNAMICS TRACKING (Clarification #1: two counters)
    # ─────────────────────────────────────────────────────────────
    prev_energy = None
    consecutive_hard = 0
    consecutive_rising = 0
    attempt = 0
    iteration = 0
    
    # ─────────────────────────────────────────────────────────────
    # MAIN CONTROL LOOP
    # ─────────────────────────────────────────────────────────────
    depth = num_recursions if num_recursions is not None else cfg.num_recursions
    while attempt < max_attempts and iteration < depth:
        attempt += 1  # ← Increments every model call
        
        try:
            # =====================================================
            # 1️⃣ GENERATION
            # =====================================================
            
            prompt_text = f"Solve step by step:\n\n{state.current}"
            
            # Use cached reasoning if available (for testing)
            cached = memory.steps.get_reasoning_by_prompt(prompt_text, state.temperature)
            
            if cached:
                reasoning = cached.reasoning_text
            else:
                reasoning = call_model(prompt_text, state.temperature)
            
            # =====================================================
            # 2️⃣ ENERGY + METRICS
            # =====================================================
            
            evidence_texts = _build_evidence(prompt, state.history)
            
            energy_result, axes, _ = gate.compute_axes(
                claim=reasoning,
                evidence_texts=evidence_texts,
            )
            
            stability_energy = _compute_stability_energy(
                claim=reasoning,
                history=state.history,
                gate=gate,
            )
            
            total_energy = axes.get("energy")
            
            all_metrics = MetricsCalculator.compute_all(
                reasoning=reasoning,
                gold_answer=gold_answer,
                energy_metrics=energy_result.to_dict(),
                cfg=cfg,
            )
            
            # Merge energy axes
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
            
            # =====================================================
            # 3️⃣ DYNAMICS COMPUTATION (global, policy-independent)
            # =====================================================
            
            # Classify violation (GLOBAL - same for all policies)
            violation = classify_violation(
                total_energy,
                cfg.tau_soft,
                cfg.tau_medium,
                cfg.tau_hard,
            )
            
            # Compute slope
            slope = compute_slope(total_energy, prev_energy)
            
            # Update counters
            consecutive_hard, consecutive_rising = update_counters(
                violation,
                slope,
                consecutive_hard,
                consecutive_rising,
                prev_energy,
            )
            
            # Check collapse (GLOBAL - containment failure)
            collapse_flag = is_collapse(
                violation,
                consecutive_hard,
                consecutive_rising,
                collapse_hard_n=getattr(cfg, "collapse_hard_n", 3),
                collapse_rising_n=getattr(cfg, "collapse_rising_n", 2),
            )
            
            # =====================================================
            # 4️⃣ COLLAPSE CHECK (Clarification #4: persist then terminate)
            # =====================================================
            
            if collapse_flag:
                # Persist snapshot WITH collapse_flag=True
                state._persist_snapshot(
                    policy_action="TERMINATE",
                    total_energy=total_energy,
                    grounding_energy=axes.get("grounding_energy"),
                    stability_energy=stability_energy,
                    energy_slope=slope,
                    violation_level=violation.value,
                    consecutive_hard=consecutive_hard,
                    consecutive_rising=consecutive_rising,
                    collapse_flag=True,
                    attempt=attempt,
                    iteration=iteration,
                )
                logger.warning(f"Collapse detected at problem {pid}, attempt {attempt}")
                break  # ← Terminate immediately
            
            # =====================================================
            # 5️⃣ POLICY DECISION
            # =====================================================
            
            dyn = EnergyDynamics(
                energy=total_energy,
                prev_energy=prev_energy,
                slope=slope,
                violation=violation,
                consecutive_hard=consecutive_hard,
                consecutive_rising=consecutive_rising,
                collapse_flag=False,
                attempt=attempt,
                iteration=iteration,
            )
            
            decision = policy.evaluate(dyn, temperature=state.temperature)
            
            # =====================================================
            # 6️⃣ APPLY DECISION
            # =====================================================
            
            if decision.action == Action.ALLOW:
                # Commit reasoning
                state.push(
                    reasoning,
                    temperature=decision.new_temperature,
                    total_energy=total_energy,
                    grounding_energy=axes.get("grounding_energy"),
                    stability_energy=stability_energy,
                    energy_slope=slope,
                    violation_level=violation.value,
                    consecutive_hard=consecutive_hard,
                    consecutive_rising=consecutive_rising,
                    collapse_flag=False,
                    attempt=attempt,
                    iteration=iteration,
                )
                prev_energy = total_energy  # ← Update only on successful commit
                iteration += 1  # ← Increments only on ACCEPT
            
            elif decision.action == Action.ROLLBACK:
                # Do NOT commit reasoning, do NOT pop (Clarification #5)
                # Just retry from current stable state with cooling
                state.temperature = decision.new_temperature
                # prev_energy stays the same (last committed energy)
                # iteration unchanged
                continue  # ← Retry same iteration
            
            elif decision.action == Action.RESET:
                # Reset to prompt (Clarification #2: clear counters)
                state.reset()
                state.temperature = decision.new_temperature
                prev_energy, consecutive_hard, consecutive_rising = reset_dynamics()
                iteration = 0  # ← Reset depth
                continue
            
            elif decision.action == Action.TERMINATE:
                state._persist_snapshot(
                    policy_action="TERMINATE",
                    total_energy=total_energy,
                    grounding_energy=axes.get("grounding_energy"),
                    stability_energy=stability_energy,
                    energy_slope=slope,
                    violation_level=violation.value,
                    consecutive_hard=consecutive_hard,
                    consecutive_rising=consecutive_rising,
                    collapse_flag=False,
                    attempt=attempt,
                    iteration=iteration,
                )
                break
        
        except Exception as e:
            logger.exception(f"❌ Crash at problem {pid}, attempt {attempt}: {e}")
            break
    
    # =====================================================
    # 7️⃣ PERSIST FINAL STEP (if not already persisted)
    # =====================================================
    
    # Note: Steps are persisted during the loop on each action
    # This ensures we capture the full trajectory including interventions


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
    cached_only: bool = False,  # ← NEW PARAM
):
    """
    Post-run tasks: aggregation, evaluation, signal discovery, cleanup.
    
    If cached_only=True, skip persistence-dependent steps since no new steps were written.
    """
    # ─────────────────────────────────────────────────────────────
    # SKIP AGGREGATION IF CACHED-ONLY (no new steps persisted)
    # ─────────────────────────────────────────────────────────────
    if cached_only:
        logger.info("⏭️ Skipping aggregation/evaluation for cached-only run: %s", run_id)
        memory.runs.update(run_id, {"status": "completed", "end_time": time.time()})
        return
    
    # 1. Metrics aggregation (defensive: handle missing steps gracefully)
    try:
        MetricsAggregator.dump_run_csv(memory, run_id)
    except Exception as e:
        logger.warning("⚠️ Aggregation skipped (no steps yet): %s", e)

    # 2. Mark run complete
    memory.runs.update(run_id, {"status": "completed", "end_time": time.time()})

    # 3. Stage 2 evaluation (defensive)
    try:
        from constrain.services.policy_evaluation_service import \
            PolicyEvaluationService
        evaluator = PolicyEvaluationService(memory)
        evaluations = evaluator.evaluate_run(run_id)
        report = evaluator.generate_report(run_id)
        logger.info("📊 Stage 2 evaluation: %s", report)
    except Exception as e:
        logger.warning("⚠️ Stage 2 evaluation skipped: %s", e)

    # 4. Signal discovery (optional)
    if cfg.run_signal_discovery:
        try:
            from constrain.services.collapse_prediction_service import \
                CollapsePredictionService
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
                    experiment_id=None,
                )
        except Exception as e:
            logger.warning("⚠️ Signal discovery skipped: %s", e)

    # 5. Populate problem_summaries (critical for utility head training)
    try:
        populate_for_run(memory, run_id)
    except Exception as e:
        logger.warning("⚠️ problem_summaries population skipped: %s", e)


def load_cached_problems(
    memory: Memory,
    seed: int,
    num_problems: int,
    min_steps: int,
) -> List[dict]:
    """
    Load only problems that have >= min_steps cached in DB.
    
    Uses filter() to get steps by prompt text, not get_by_prompt (which returns str).
    """
    from datasets import load_dataset
    
    dataset = load_dataset("gsm8k", "main", split="test")
    dataset = dataset.shuffle(seed=seed)
    
    cached_examples = []
    
    for pid, example in enumerate(dataset):
        prompt = example["question"]
        
        # ✅ FIX: Use filter to get steps by prompt, then count them
        # get_by_prompt returns str (reasoning text), not list of steps
        steps = memory.steps.filter(run_id=None)  # Get all steps, filter in Python
        matching_steps = [s for s in steps if s.prompt_text == prompt]
        
        if len(matching_steps) >= min_steps:
            cached_examples.append(example)
        
        if len(cached_examples) >= num_problems:
            break
    
    logger.info(f"Found {len(cached_examples)}/{num_problems} problems with cached steps")
    return cached_examples