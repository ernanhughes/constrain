import logging
import random
import time
import uuid

import numpy as np
from datasets import load_dataset
from tqdm.auto import tqdm

from constrain.analysis.aggregation.metrics_aggregator import MetricsAggregator
from constrain.analysis.aggregation.metrics_calculator import MetricsCalculator
from constrain.analysis.stage3.signal_discovery_service import \
    SignalDiscoveryService
from constrain.analysis.visualization.dashboard_exporter import \
    DashboardExporter
from constrain.config import get_config
from constrain.data.memory import Memory
from constrain.data.schemas.intervention import InterventionDTO
from constrain.data.schemas.run import RunDTO
from constrain.data.schemas.step import StepDTO
from constrain.energy.embedding.hf_embedder import HFEmbedder
from constrain.energy.embedding.sqlite_embedding_backend import \
    SQLiteEmbeddingBackend
from constrain.energy.gate import VerifiabilityGate
from constrain.energy.geometry.claim_evidence import ClaimEvidenceGeometry
from constrain.energy.utils.text_utils import split_into_sentences
from constrain.model import call_model
from constrain.policy.apply_policy import apply_policy
from constrain.reasoning_state import ReasoningState
from constrain.utils.dict_utils import flatten_numeric_dict

logger = logging.getLogger(__name__)


def run(policy_id: int = 4, seed: int = 42, num_problems: int = None, threshold: float = None) -> str:

    start_time = time.time()
    logger.debug("=" * 60)
    logger.debug("Starting run with policy_id=%d, seed=%d", policy_id, seed)
    logger.debug("=" * 60)

    # ==========================================================
    # STAGE 0 — INIT
    # ==========================================================

    random.seed(seed)
    np.random.seed(seed)

    cfg = get_config()
    logger.debug("Configuration loaded: embedding_model=%s, num_problems=%d, num_recursions=%d",
                cfg.embedding_model, cfg.num_problems, cfg.num_recursions)

    memory = Memory(cfg.db_url)
    logger.debug("Memory initialized with DB: %s", cfg.db_url)

    embedder = HFEmbedder(
        model_name=cfg.embedding_model,
        backend=SQLiteEmbeddingBackend(str(cfg.embedding_db)),
    )
    logger.debug("Embedder initialized (model=%s, backend=%s)", cfg.embedding_model, cfg.embedding_db)

    energy_computer = ClaimEvidenceGeometry(top_k=6, rank_r=4)
    logger.debug("Energy computer configured: top_k=6, rank_r=4")

    gate = VerifiabilityGate(
        embedder=embedder,
        energy_computer=energy_computer,
    )
    logger.debug("VerifiabilityGate created")

    run_id = f"run_{uuid.uuid4().hex[:8]}"
    logger.debug(f"Run ID: {run_id}")

    # ==========================================================
    # REGISTER RUN
    # ==========================================================

    run_dto = RunDTO(
        run_id=run_id,
        model_name=cfg.model_name,
        initial_temperature=cfg.initial_temperature,
        num_problems=num_problems or cfg.num_problems,
        num_recursions=cfg.num_recursions,
        tau_soft=cfg.tau_soft,
        tau_medium=cfg.tau_medium,
        tau_hard=cfg.tau_hard,
        policy_id=policy_id,
        task_type="gsm8k",
        start_time=start_time,
        status="running",
        notes=cfg.notes,
        seed=seed,
    )
    memory.runs.create(run_dto)
    logger.debug("Run registered in database")

    # ==========================================================
    # LOAD DATASET
    # ==========================================================

    logger.debug("Loading GSM8K dataset (test split)...")
    dataset = load_dataset("gsm8k", "main", split="test")
    logger.debug("Dataset loaded, total examples: %d", len(dataset))

    dataset = dataset.shuffle(seed=seed).select(range(cfg.num_problems))
    logger.debug("Selected %d problems after shuffle", cfg.num_problems)

    # ==========================================================
    # MAIN LOOP
    # ==========================================================

    logger.debug("Starting main loop over problems")
    total_problems = len(dataset)
    for pid, example in enumerate(tqdm(dataset, desc="Problems")):

        prompt = example["question"]
        gold_answer = example["answer"].split("####")[-1].strip()
        logger.debug("Problem %d/%d: prompt length=%d, gold=%s",
                     pid+1, total_problems, len(prompt), gold_answer[:50])

        state = ReasoningState(prompt)
        state.temperature = cfg.initial_temperature
        logger.debug("Initialized reasoning state with temperature=%.2f", state.temperature)

        for iteration in range(cfg.num_recursions):

            try:
                prompt_text = f"Solve step by step:\n\n{state.current}"
                temperature = state.temperature
                logger.debug("Iteration %d: temperature=%.2f, current state length=%d",
                             iteration, temperature, len(state.current))

                # -----------------------------
                # Model
                # -----------------------------

                cached = memory.steps.get_reasoning_by_prompt(prompt, temperature)
                if cached:
                    reasoning = cached.reasoning_text
                    logger.debug("Cache HIT for prompt (temp=%.2f)", temperature)
                else:
                    logger.debug("Cache MISS, calling model (temp=%.2f)", temperature)
                    reasoning = call_model(prompt_text, temperature)
                    logger.debug("Model returned reasoning (length=%d)", len(reasoning))

                # -----------------------------
                # Build Evidence
                # -----------------------------

                evidence_texts = []
                evidence_texts.extend(split_into_sentences(prompt))
                for past in state.history:
                    evidence_texts.extend(split_into_sentences(past))
                logger.debug("Built evidence: %d sentences total", len(evidence_texts))

                # -----------------------------
                # Energy (Gate)
                # -----------------------------

                energy_result, axes, _ = gate.compute_axes(
                    claim=reasoning,
                    evidence_texts=evidence_texts,
                )
                logger.debug("Energy result: energy=%.4f, participation_ratio=%.4f, sensitivity=%.4f, alignment=%.4f",
                             axes.get("energy"), axes.get("participation_ratio"),
                             axes.get("sensitivity"), axes.get("alignment"))

                # Stability energy (previous accepted step only)
                if state.history:
                    last = state.history[-1]
                    stability_result, stability_axes, _ = gate.compute_axes(
                        claim=reasoning,
                        evidence_texts=split_into_sentences(last),
                    )
                    stability_energy = stability_axes.get("energy")
                    logger.debug("Stability energy (vs last accepted): %.4f", stability_energy)
                else:
                    stability_energy = 0.0

                total_energy = axes.get("energy")


                # -----------------------------
                # METRICS
                # -----------------------------

                all_metrics = MetricsCalculator.compute_all(
                    reasoning=reasoning,
                    gold_answer=gold_answer,
                    energy_metrics=energy_result.to_dict(),
                    cfg=cfg,
                )
                logger.debug("Metrics computed: correctness=%s, accuracy=%.3f, phase=%s",
                             all_metrics.get("correctness"), all_metrics.get("accuracy"),
                             all_metrics.get("phase_label"))

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


                # -----------------------------
                # POLICY
                # -----------------------------

                action, new_temperature, collapse_probability = apply_policy(
                    policy_id=policy_id,
                    axes=axes,
                    flat_metrics=flat_metrics,
                    reasoning=reasoning,
                    state=state,
                    memory=memory,
                    run_id=run_id,
                    threshold=threshold,
                )
                logger.debug("Policy decision: action=%s, new_temperature=%.2f", action, new_temperature)

                # Apply state transition
                if action == "ACCEPT":
                    state.accept(reasoning)
                    logger.debug("State accepted, history length now %d", len(state.history))

                elif action == "REVERT":
                    state.revert()
                    logger.debug("State reverted, history length now %d", len(state.history))

                elif action in ["RESET", "RESET_PROMPT"]:
                    state.reset()
                    logger.debug("State reset to original prompt")

                state.temperature = new_temperature

                # -----------------------------
                # SAVE STEP
                # -----------------------------

                step_dto = StepDTO(
                    run_id=run_id,
                    problem_id=pid,
                    iteration=iteration,
                    prompt_text=prompt,
                    reasoning_text=reasoning,
                    collapse_probability=collapse_probability,
                    gold_answer=gold_answer,
                    extracted_answer=all_metrics["extracted_answer"],
                    total_energy=total_energy,
                    grounding_energy=energy_result.energy,
                    stability_energy=stability_energy,
                    correctness=all_metrics.get("correctness"),
                    accuracy=all_metrics.get("accuracy"),
                    temperature=state.temperature,
                    policy_action=action,
                    phase=MetricsCalculator.PHASE_VALUE_TO_LABEL[
                        all_metrics["phase_value"]
                    ],
                    timestamp=time.time(),
                )

                step_dto = memory.steps.create(step_dto)
                logger.debug("Step saved with id=%s", step_dto.id)

                memory.metrics.bulk_from_dict(
                    step_id=step_dto.id,
                    stage="energy_v2",
                    metrics=flat_metrics,
                )
                logger.debug("Metrics saved for step")

                # -----------------------------
                # INTERVENTION LOG
                # -----------------------------

                if action != "ACCEPT":
                    intervention = InterventionDTO(
                        run_id=run_id,
                        problem_id=pid,
                        iteration=iteration,
                        threshold="learned",
                        rationale=action,
                        reverted_to=iteration - 1,
                        new_temperature=new_temperature,
                        timestamp=time.time(),
                    )
                    memory.interventions.create(intervention)
                    logger.debug("Intervention recorded: %s at iter %d", action, iteration)

            except Exception as e:
                logger.exception(f"Crash at problem {pid}, iteration {iteration}: {e}")
                break

    # ==========================================================
    # AGGREGATION
    # ==========================================================

    try:
        logger.debug("Starting metrics aggregation for run %s", run_id)
        MetricsAggregator.dump_run_csv(memory, run_id)
        logger.debug("CSV dump completed")
    except Exception as e:
        logger.exception(f"Aggregation failed: {e}")

    # ==========================================================
    # FINALIZE
    # ==========================================================

    memory.runs.update(
        run_id,
        {
            "status": "completed",
            "end_time": time.time(),
        },
    )
    logger.debug("Run marked as completed in database")

    # ==========================================================
    # SIGNAL DISCOVERY
    # ==========================================================

    try:
        run_info = memory.runs.get_by_id(run_id)
        if run_info.status == "completed":
            logger.debug("Starting signal discovery for run %s", run_id)
            service = SignalDiscoveryService(memory)
            results = service.analyze_and_persist(run_id)
            logger.debug("Signal discovery completed, exporting dashboards")
            DashboardExporter.export_json(results, run_id)
            DashboardExporter.export_html(results, run_id)
            logger.debug("Dashboards exported")
    except Exception as e:
        logger.exception(f"Signal discovery failed: {e}")

    elapsed = time.time() - start_time
    logger.debug("=" * 60)
    logger.debug("Run %s finished in %.2f seconds", run_id, elapsed)
    logger.debug("=" * 60)

    return run_id