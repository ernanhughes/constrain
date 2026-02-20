import uuid
import time
from datasets import load_dataset
from tqdm.auto import tqdm

from constrain.data.memory import Memory
from constrain.config import get_config
from constrain.policy import apply_policy
from constrain.model import call_model
from constrain.energy_computer import compute_energy
from constrain.analysis.stage3.rolling_monitor import RollingSignalMonitor

from constrain.data.schemas.run import RunDTO
from constrain.data.schemas.step import StepDTO


def run_live(policy_id: int = 5):

    cfg = get_config()
    memory = Memory()

    run_id = f"live_{uuid.uuid4().hex[:8]}"

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
        notes="Live signal monitoring run",
    )

    memory.runs.create(run_dto)

    dataset = load_dataset("gsm8k", "main", split="test").select(
        range(cfg.num_problems)
    )

    monitor = RollingSignalMonitor(memory, interval_steps=50)

    pbar = tqdm(dataset, desc="Live Problems")

    for pid, example in enumerate(pbar):

        prompt = example["question"]
        state = prompt
        temperature = cfg.initial_temperature
        prev_reasoning = None

        for iteration in range(cfg.num_recursions):

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
                state,
                prompt,
                temperature,
                memory,
                run_id,
            )

            step_dto = StepDTO(
                run_id=run_id,
                problem_id=pid,
                iteration=iteration,
                prompt_text=prompt,
                reasoning_text=reasoning,
                gold_answer=None,
                extracted_answer=None,
                total_energy=energy_metrics["total_energy"],
                grounding_energy=energy_metrics["grounding_energy"],
                stability_energy=energy_metrics["stability_energy"],
                temperature=temperature,
                policy_action=action,
                phase="live",
                timestamp=time.time(),
            )

            memory.steps.create(step_dto)

            prev_reasoning = reasoning
            state = new_state

            # 🔥 Live signal check
            monitor.maybe_run(run_id)

    memory.runs.update(
        run_id,
        {
            "status": "completed",
            "end_time": time.time(),
        },
    )

    print(f"✅ Live run complete: {run_id}")
