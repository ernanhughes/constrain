import uuid
import time
from datasets import load_dataset
import random
import numpy as np
from constrain.data.memory import Memory
from constrain.config import get_config
from constrain.policy import apply_policy
from constrain.model import call_model
from constrain.energy_computer import compute_energy

from constrain.data.schemas.run import RunDTO
from constrain.data.schemas.step import StepDTO
from constrain.data.schemas.intervention import InterventionDTO
from constrain.analysis.aggregation.metrics_calculator import MetricsCalculator
from tqdm.auto import tqdm
from constrain.analysis.aggregation.metrics_aggregator import MetricsAggregator
from constrain.analysis.visualization.dashboard_exporter import DashboardExporter
from constrain.analysis.stage3.signal_discovery_service import SignalDiscoveryService

import logging
logger = logging.getLogger(__name__)

def run(policy_id: int = 4, seed: int = 42) -> str:
    """
    Main execution pipeline.

    Returns:
        run_id (str)
    """

    start_time = time.time()

    # ==========================================================
    # STAGE 0 — Initialization
    # ==========================================================

    logger.info("========== STAGE 0: INITIALIZATION ==========")

    random.seed(seed)
    np.random.seed(seed)

    cfg = get_config()
    memory = Memory()

    run_id = f"run_{uuid.uuid4().hex[:8]}"
    logger.info(f"Run ID: {run_id}")
    logger.info(f"Policy ID: {policy_id}")
    logger.info(f"Seed: {seed}")

    # ==========================================================
    # STAGE 1 — Register Run
    # ==========================================================

    logger.info("========== STAGE 1: REGISTER RUN ==========")

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
    # STAGE 2 — Load Dataset
    # ==========================================================

    logger.info("========== STAGE 2: DATASET ==========")

    dataset = load_dataset("gsm8k", "main", split="test")

    if cfg.fast_mode:
        logger.info("⚡ FAST MODE ENABLED")

        existing_prompts = set(memory.steps.get_distinct_prompts())
        fast_examples = []
        new_examples = []

        for ex in dataset:
            if ex["question"] in existing_prompts:
                fast_examples.append(ex)
            else:
                new_examples.append(ex)

        logger.info(
            f"Dataset split → Cached: {len(fast_examples)}, New: {len(new_examples)}"
        )

        dataset = (fast_examples + new_examples)[: cfg.num_problems]
    else:
        dataset = dataset.shuffle(seed=0).select(range(cfg.num_problems))

    # ==========================================================
    # STAGE 3 — Main Execution Loop
    # ==========================================================

    logger.info("========== STAGE 3: EXECUTION LOOP ==========")
    pbar = tqdm(dataset, desc="Problems", total=cfg.num_problems)

    for pid, example in enumerate(pbar):

        prompt = example["question"]
        gold_answer = example["answer"].split("####")[-1].strip()

        state = prompt
        last_stable = prompt
        reasoning_history = []
        temperature = cfg.initial_temperature
        prev_reasoning = None

        for iteration in range(cfg.num_recursions):

            try:
                prompt_text = f"Solve step by step:\n\n{prompt}"

                # -----------------------------
                # Model (with caching)
                # -----------------------------

                cached = memory.steps.get_reasoning_by_prompt(prompt, temperature)

                if cached:
                    reasoning = cached.reasoning_text
                    logger.debug(f"[Cache hit] P{pid} I{iteration}")
                else:
                    reasoning = call_model(prompt_text, temperature)

                # -----------------------------
                # Energy
                # -----------------------------

                energy_metrics = compute_energy(
                    prompt=prompt,
                    current=reasoning,
                    reasoning_history=reasoning_history,
                )

                # -----------------------------
                # Policy
                # -----------------------------

                new_state, temperature, action = apply_policy(
                    policy_id,
                    energy_metrics["total_energy"],
                    reasoning,
                    last_stable,
                    prompt,
                    temperature,
                    memory,
                    run_id=run_id,
                )

                if action == "ACCEPT":
                    last_stable = reasoning
                    reasoning_history.append(reasoning)

                # -----------------------------
                # Metrics
                # -----------------------------

                all_metrics = MetricsCalculator.compute_all(
                    reasoning=reasoning,
                    gold_answer=gold_answer,
                    energy_metrics=energy_metrics,
                    cfg=cfg,
                )

                # -----------------------------
                # Persist Step
                # -----------------------------

                step_dto = StepDTO(
                    run_id=run_id,
                    problem_id=pid,
                    iteration=iteration,
                    prompt_text=prompt,
                    reasoning_text=reasoning,
                    gold_answer=gold_answer,
                    extracted_answer=all_metrics["extracted_answer"],
                    total_energy=energy_metrics["total_energy"],
                    grounding_energy=energy_metrics["grounding_energy"],
                    stability_energy=energy_metrics["stability_energy"],
                    correctness=all_metrics.get("correctness"),
                    accuracy=all_metrics.get("accuracy"),
                    temperature=temperature,
                    policy_action=action,
                    phase=MetricsCalculator.PHASE_VALUE_TO_LABEL[
                        all_metrics["phase_value"]
                    ],
                    timestamp=time.time(),
                )

                step_dto = memory.steps.create(step_dto)

                memory.metrics.bulk_from_dict(
                    step_id=step_dto.id,
                    stage="energy_v1",
                    metrics=all_metrics,
                )

                # -----------------------------
                # Intervention
                # -----------------------------

                if action != "ACCEPT":
                    memory.interventions.create(
                        InterventionDTO(
                            run_id=run_id,
                            problem_id=pid,
                            iteration=iteration,
                            threshold="dynamic",
                            rationale=action,
                            reverted_to=iteration - 1,
                            new_temperature=temperature,
                            timestamp=time.time(),
                        )
                    )

            except Exception as e:
                print("Error during execution loop:", e)
                logger.exception(
                    f"Crash at problem {pid}, iteration {iteration}: {e}"
                )
                break

    # ==========================================================
    # STAGE 4 — Aggregation
    # ==========================================================

    logger.info("========== STAGE 4: AGGREGATION ==========")

    try:
        MetricsAggregator.dump_run_csv(memory, run_id)
    except Exception as e:
        logger.exception(f"Aggregation failed: {e}")

    # ==========================================================
    # STAGE 5 — Finalize Run
    # ==========================================================

    logger.info("========== STAGE 5: FINALIZE ==========")

    memory.runs.update(
        run_id,
        {
            "status": "completed",
            "end_time": time.time(),
        },
    )

    # ==========================================================
    # STAGE 6 — Signal Discovery
    # ==========================================================

    logger.info("========== STAGE 6: SIGNAL DISCOVERY ==========")

    try:
        # ✅ Check run completed successfully
        run_info = memory.runs.get_by_id(run_id)
        if run_info.status != "completed":
            logger.warning(f"⚠️ Skipping signal discovery: run status = {run_info.status}")
        else:
            service = SignalDiscoveryService(memory)
            results = service.analyze_and_persist(run_id)
            DashboardExporter.export_json(results, run_id)
            DashboardExporter.export_html(results, run_id)
            logger.info("Signal discovery complete")

    except Exception as e:
        logger.exception(f"Signal discovery failed: {e}")

    elapsed = time.time() - start_time
    logger.info(f"Run completed in {elapsed:.2f}s")

    return run_id