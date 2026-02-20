import uuid
import time
from datasets import load_dataset

from constrain.data.memory import Memory
from constrain.config import get_config
from constrain.policy import apply_policy
from constrain.model import call_model
from constrain.energy import compute_energy

from constrain.data.schemas.run import RunDTO
from constrain.data.schemas.step import StepDTO
from constrain.data.schemas.intervention import InterventionDTO
from constrain.analysis.metrics_calculator import MetricsCalculator
from tqdm.auto import tqdm
from constrain.analysis.metrics_aggregator import MetricsAggregator
from constrain.analysis.dashboard_exporter import DashboardExporter
from constrain.analysis.signal_discovery_service import SignalDiscoveryService

def extract_number(text):
    import re
    nums = re.findall(r"-?\d+\.?\d*", text)
    return float(nums[-1]) if nums else None


def run(policy_id: int = 4):

    cfg = get_config()
    memory = Memory()

    # -------------------------------------------------
    # Create Run
    # -------------------------------------------------

    run_id = f"run_{uuid.uuid4().hex[:8]}"

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
        start_time=time.time(),
        status="running",
        notes=cfg.notes,
    )

    memory.runs.create(run_dto)

    # -------------------------------------------------
    # Dataset
    # -------------------------------------------------

    dataset = load_dataset("gsm8k", "main", split="test").select(
        range(cfg.num_problems)
    )

    # -------------------------------------------------
    # Main Loop
    # -------------------------------------------------


    pbar = tqdm(dataset, desc="Problems", total=cfg.num_problems)

    for pid, example in enumerate(pbar):

        prompt = example["question"]
        gold_answer = example["answer"].split("####")[-1].strip()

        state = prompt
        last_stable = prompt
        temperature = cfg.initial_temperature
        prev_reasoning = None

        for iteration in range(cfg.num_recursions):

            try:

                reasoning = call_model(
                    f"Solve step by step:\n\n{state}",
                    temperature,
                )

                energy_metrics = compute_energy(
                    memory,
                    prompt,
                    reasoning,
                    prev_reasoning,
                )

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

                # -------------------------------
                # Compute All Metrics
                # -------------------------------

                all_metrics = MetricsCalculator.compute_all(
                    reasoning=reasoning,
                    gold_answer=gold_answer,
                    energy_metrics=energy_metrics,
                    cfg=cfg,
                )

                # -------------------------------
                # Store Step
                # -------------------------------

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
                    temperature=temperature,
                    policy_action=action,
                    phase=MetricsCalculator.PHASE_VALUE_TO_LABEL[all_metrics["phase_value"]],
                    timestamp=time.time(),
                )

                step_dto = memory.steps.create(step_dto)

                # -------------------------------
                # Store Metrics (ALL of them)
                # -------------------------------

                memory.metrics.bulk_from_dict(
                    step_id=step_dto.id,
                    stage="energy_v1",
                    metrics=all_metrics,
                )

                # -------------------------------
                # Intervention
                # -------------------------------

                if action != "ACCEPT":

                    intervention_dto = InterventionDTO(
                        run_id=run_id,
                        problem_id=pid,
                        iteration=iteration,
                        threshold="dynamic",
                        rationale=action,
                        reverted_to=iteration - 1,
                        new_temperature=temperature,
                        timestamp=time.time(),
                    )

                    memory.interventions.create(intervention_dto)

                prev_reasoning = reasoning
                state = new_state

            except Exception as e:
                print(f"⚠ Crash at problem {pid}, iter {iteration}: {e}")
                break


    df = MetricsAggregator.build_run_dataframe(memory, run_id)
    print(df.head())
    MetricsAggregator.dump_run_csv(memory, run_id)

    # -------------------------------------------------
    # Finish Run
    # -------------------------------------------------

    memory.runs.update(
        run_id,
        {
            "status": "completed",
            "end_time": time.time(),
        },
    )

    print(f"✅ Run complete: {run_id}")

    # -------------------------------------------------
    # Signal Discovery (Post-Run Phase)
    # -------------------------------------------------

    try:
        print("🔍 Running signal discovery...")

        service = SignalDiscoveryService(memory)
        results = service.analyze_and_persist(run_id)

        DashboardExporter.export_json(results, run_id)
        DashboardExporter.export_html(results, run_id)

        print("📊 Signal discovery complete")

    except Exception as e:
        print(f"⚠ Signal discovery failed: {e}")
