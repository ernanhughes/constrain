from __future__ import annotations

from datetime import datetime
from typing import List, Optional

import pandas as pd

from constrain.config import get_config
from constrain.data.memory import Memory
from constrain.data.schemas.experiment import ExperimentDTO
from constrain.core.runner import run
from constrain.services.policy_evaluation_service import \
    PolicyEvaluationService


def compare_policies(
    policy_ids: List[int],
    seeds: tuple[int, ...] = (42, 43, 44),
    num_problems: Optional[int] = None,
    num_recursions: Optional[int] = None,
    experiment_name: Optional[str] = None,
    initial_temperature: Optional[float] = None,
) -> pd.DataFrame:
    """
    Run policy comparison experiment with automatic evaluation.
    """
    cfg = get_config()
    memory = Memory(cfg.db_url)
    evaluator = PolicyEvaluationService(memory)

    # Create experiment record
    if experiment_name is None:
        experiment_name = f"policy_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    experiment_dto = ExperimentDTO.create(
        experiment_name=experiment_name,
        experiment_type="policy_comparison",
        policy_ids=policy_ids,
        seeds=list(seeds),
        num_problems=num_problems or cfg.num_problems,
        num_recursions=num_recursions or cfg.num_recursions,
        notes=f"Comparing policies {policy_ids} across seeds {seeds}",
        initial_temperature=initial_temperature,
    )
    experiment = memory.experiments.create(experiment_dto)
    experiment_id = experiment.id

    print(f"\n{'='*60}")
    print(f"Experiment ID: {experiment_id}")
    print(f"Experiment Name: {experiment_name}")
    print(f"{'='*60}")

    all_rows = []
    run_ids = []

    for seed in seeds:
        print(f"\n{'='*60}")
        print(f"Running seed {seed}")
        print(f"{'='*60}")

        for pid in policy_ids:
            print(f"\n🚀 Policy {pid} (seed={seed})")

            try:
                # Run experiment
                run_id = run(
                    policy_id=pid,
                    seed=seed,
                    num_problems=num_problems,
                    num_recursions=num_recursions,
                    initial_temperature=initial_temperature,
                )
                run_ids.append(run_id)

                # Associate run with experiment (update run record)
                memory.runs.update(run_id, {"experiment_id": experiment_id})

                # Evaluate and persist
                summary, problem_df = evaluator.evaluate_run(run_id)

                # Generate report (includes historical comparison)
                report = evaluator.generate_report(run_id)
                print(f"\n{report}")

                # Collect summary for DataFrame
                all_rows.append({
                    "run_id": run_id,
                    "policy_id": pid,
                    "seed": seed,
                    "experiment_id": experiment_id,
                    "accuracy": summary.accuracy,
                    "intervention_rate": summary.intervention_rate,
                    "collapse_rate": summary.collapse_rate,
                    "avg_energy": summary.avg_energy,
                    "avg_recursions": summary.avg_recursions,
                    "intervention_help_rate": summary.intervention_help_rate,
                    "intervention_harm_rate": summary.intervention_harm_rate,
                    "num_problems": summary.num_problems,
                })


            except Exception as e:
                print(f"⚠ Run failed for policy {pid}, seed {seed}: {e}")
                continue

    # Build results DataFrame
    if not all_rows:
        print("❌ No successful runs.")
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)

    # Statistical comparisons
    print(f"\n{'='*60}")
    print("STATISTICAL COMPARISONS (vs Baseline)")
    print(f"{'='*60}")

    results = evaluator.compare_policies(policy_ids=policy_ids, experiment_id=experiment_id)

    for pid, result in results.items():
        sig_marker = "✅" if result.significant else "⚠️"
        print(f"\nPolicy {pid} vs Baseline:")
        print(f"  Δ Accuracy: {result.mean_diff*100:+.2f}%")
        print(f"  95% CI: [{result.ci_lower*100:+.2f}%, {result.ci_upper*100:+.2f}%]")
        print(f"  {sig_marker} {'Significant' if result.significant else 'Not significant'}")

    # Complete experiment with summary
    experiment_summary = {
        "n_runs": len(run_ids),
        "n_problems_total": int(df["num_problems"].sum()),
        "accuracy_by_policy": df.groupby("policy_id")["accuracy"].mean().to_dict(),
        "intervention_rate_by_policy": df.groupby("policy_id")["intervention_rate"].mean().to_dict(),
        "statistical_results": {
            str(pid): {
                "mean_diff": r.mean_diff,
                "ci_lower": r.ci_lower,
                "ci_upper": r.ci_upper,
                "significant": r.significant,
            }
            for pid, r in results.items()
        },
    }
    memory.experiments.complete(experiment_id, experiment_summary)

    # Save results
    output_path = f"{get_config().reports_dir}/policy_comparison_{experiment_id}.csv"
    df.to_csv(output_path, index=False)
    print(f"\n✅ Results saved to {output_path}")
    print(f"✅ Experiment {experiment_id} marked as completed")

    return df


if __name__ == "__main__":
    df = compare_policies(
        policy_ids=[0, 4, 99],
        seeds=(42, 43, 44),
        num_problems=20,
    )