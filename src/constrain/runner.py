import uuid
import time
import random
import logging
import numpy as np
from datasets import load_dataset
from tqdm.auto import tqdm

from constrain.data.memory import Memory
from constrain.config import get_config
from constrain.model import call_model
from constrain.policy import apply_policy
from constrain.reasoning_state import ReasoningState
from constrain.energy.utils.text_utils import split_into_sentences

from constrain.data.schemas.run import RunDTO
from constrain.data.schemas.step import StepDTO
from constrain.data.schemas.intervention import InterventionDTO

from constrain.analysis.aggregation.metrics_calculator import MetricsCalculator
from constrain.analysis.aggregation.metrics_aggregator import MetricsAggregator
from constrain.analysis.visualization.dashboard_exporter import DashboardExporter
from constrain.analysis.stage3.signal_discovery_service import SignalDiscoveryService

from constrain.energy.geometry.claim_evidence import ClaimEvidenceGeometry
from constrain.energy.embedding.hf_embedder import HFEmbedder
from constrain.energy.embedding.sqlite_embedding_backend import SQLiteEmbeddingBackend
from constrain.energy.gate import VerifiabilityGate
from constrain.utils.dict_utils import flatten_numeric_dict

logger = logging.getLogger(__name__)


def run(policy_id: int = 4, seed: int = 42) -> str:

    start_time = time.time()

    # ==========================================================
    # STAGE 0 — INIT
    # ==========================================================

    random.seed(seed)
    np.random.seed(seed)

    cfg = get_config()
    memory = Memory(cfg.db_url)

    embedder = HFEmbedder(
        model_name=cfg.embedding_model,
        backend=SQLiteEmbeddingBackend(str(cfg.embedding_db)),
    )

    energy_computer = ClaimEvidenceGeometry(top_k=6, rank_r=4)

    gate = VerifiabilityGate(
        embedder=embedder,
        energy_computer=energy_computer,
    )

    run_id = f"run_{uuid.uuid4().hex[:8]}"
    logger.info(f"Run ID: {run_id}")

    # ==========================================================
    # REGISTER RUN
    # ==========================================================

    run_dto = RunDTO(
        run_id=run_id,
        model_name=cfg.model_name,
        initial_temperature=cfg.initial_temperature,
        num_problems=cfg.num_problems,
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

    # ==========================================================
    # LOAD DATASET
    # ==========================================================

    dataset = load_dataset("gsm8k", "main", split="test")
    dataset = dataset.shuffle(seed=seed).select(range(cfg.num_problems))

    # ==========================================================
    # MAIN LOOP
    # ==========================================================

    for pid, example in enumerate(tqdm(dataset, desc="Problems")):

        prompt = example["question"]
        gold_answer = example["answer"].split("####")[-1].strip()

        state = ReasoningState(prompt)
        state.temperature = cfg.initial_temperature

        for iteration in range(cfg.num_recursions):

            try:
                prompt_text = f"Solve step by step:\n\n{state.current}"
                temperature = state.temperature

                # -----------------------------
                # Model
                # -----------------------------

                cached = memory.steps.get_reasoning_by_prompt(prompt, temperature)

                if cached:
                    reasoning = cached.reasoning_text
                else:
                    reasoning = call_model(prompt_text, temperature)

                # -----------------------------
                # Build Evidence
                # -----------------------------

                evidence_texts = []
                evidence_texts.extend(split_into_sentences(prompt))

                for past in state.history:
                    evidence_texts.extend(split_into_sentences(past))

                # -----------------------------
                # Energy (Gate)
                # -----------------------------

                energy_result, axes, _ = gate.compute_axes(
                    claim=reasoning,
                    evidence_texts=evidence_texts,
                )

                # Stability energy (previous accepted step only)
                if state.history:
                    last = state.history[-1]
                    stability_result, stability_axes, _ = gate.compute_axes(
                        claim=reasoning,
                        evidence_texts=split_into_sentences(last),
                    )
                    stability_energy = stability_axes.get("energy")
                else:
                    stability_energy = 0.0

                total_energy = axes.get("energy")

                # -----------------------------
                # POLICY
                # -----------------------------

                action, new_temperature = apply_policy(
                    policy_id=policy_id,
                    axes=axes,
                    reasoning=reasoning,
                    state=state,
                    memory=memory,
                    run_id=run_id,
                )

                # Apply state transition
                if action == "ACCEPT":
                    state.accept(reasoning)

                elif action == "REVERT":
                    state.revert()

                elif action in ["RESET", "RESET_PROMPT"]:
                    state.reset()

                state.temperature = new_temperature

                # -----------------------------
                # METRICS
                # -----------------------------

                all_metrics = MetricsCalculator.compute_all(
                    reasoning=reasoning,
                    gold_answer=gold_answer,
                    energy_metrics=energy_result.to_dict(),
                    cfg=cfg,
                )

                all_metrics.update({
                    "energy": axes.get("energy"),
                    "participation_ratio": axes.get("participation_ratio"),
                    "sensitivity": axes.get("sensitivity"),
                    "alignment": axes.get("alignment"),
                    "sim_margin": axes.get("sim_margin"),
                    "iteration": iteration,
                    "evidence_count": len(state.history),
                })

                # -----------------------------
                # SAVE STEP
                # -----------------------------

                step_dto = StepDTO(
                    run_id=run_id,
                    problem_id=pid,
                    iteration=iteration,
                    prompt_text=prompt,
                    reasoning_text=reasoning,
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

                flat_metrics = flatten_numeric_dict(all_metrics)
                memory.metrics.bulk_from_dict(
                    step_id=step_dto.id,
                    stage="energy_v2",
                    metrics=flat_metrics,
                )


                # -----------------------------
                # INTERVENTION LOG
                # -----------------------------

                if action != "ACCEPT":
                    memory.interventions.create(
                        InterventionDTO(
                            run_id=run_id,
                            problem_id=pid,
                            iteration=iteration,
                            threshold="learned",
                            rationale=action,
                            reverted_to=iteration - 1,
                            new_temperature=new_temperature,
                            timestamp=time.time(),
                        )
                    )

            except Exception as e:
                logger.exception(f"Crash at problem {pid}, iteration {iteration}: {e}")
                break

    # ==========================================================
    # AGGREGATION
    # ==========================================================

    try:
        MetricsAggregator.dump_run_csv(memory, run_id)
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

    # ==========================================================
    # SIGNAL DISCOVERY
    # ==========================================================

    try:
        run_info = memory.runs.get_by_id(run_id)
        if run_info.status == "completed":
            service = SignalDiscoveryService(memory)
            results = service.analyze_and_persist(run_id)
            DashboardExporter.export_json(results, run_id)
            DashboardExporter.export_html(results, run_id)
    except Exception as e:
        logger.exception(f"Signal discovery failed: {e}")

    logger.info(f"Run completed in {time.time() - start_time:.2f}s")

    return run_id
