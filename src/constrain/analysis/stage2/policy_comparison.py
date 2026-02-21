import numpy as np
import pandas as pd

from constrain.analysis.stage2.application_evaluator import \
    ApplicationEvaluator
from constrain.config import get_config
from constrain.data.memory import Memory
from constrain.runner import run


def bootstrap_diff(a, b, n=2000, seed=42):
    rng = np.random.RandomState(seed)
    diffs = []

    a = np.array(a)
    b = np.array(b)

    for _ in range(n):
        idx = rng.choice(len(a), len(a), replace=True)
        diffs.append(a[idx].mean() - b[idx].mean())

    lower = np.percentile(diffs, 2.5)
    upper = np.percentile(diffs, 97.5)

    return {
        "mean_diff": float(np.mean(diffs)),
        "ci_lower": float(lower),
        "ci_upper": float(upper),
    }


def compare_policies(policy_ids, seeds=(42, 43, 44)):

    all_results = []

    for seed in seeds:

        print("\n==============================")
        print(f"Running seed {seed}")
        print("==============================")

        for pid in policy_ids:

            print(f"\n🚀 Policy {pid} (seed={seed})")

            run_id = run(policy_id=pid, seed=seed)

            memory = Memory(get_config().db_url)
            evaluator = ApplicationEvaluator(memory)

            summary, problem_df = evaluator.evaluate_run(run_id)

            summary["policy_id"] = pid
            summary["seed"] = seed

            # Store per-problem correctness for bootstrap
            summary["_per_problem_correct"] = problem_df["final_correct"].values

            steps = memory.steps.get_by_run(run_id)
    

            steps_df = pd.DataFrame([s.model_dump() for s in steps])

            collapsed_per_problem = (
                steps_df.groupby("problem_id")["phase"]
                .apply(lambda x: (x == "collapse").any())
            )

            problem_df["collapsed"] = problem_df["problem_id"].map(collapsed_per_problem)
            problem_df["collapsed"] = problem_df["collapsed"].fillna(False) 



            all_results.append(summary)

    df = pd.DataFrame(all_results)

    print("\n==============================")
    print("Raw Results")
    print("==============================")
    print(df.drop(columns=["_per_problem_correct"]))

    # ---------------------------------------------------------
    # Statistical Comparison
    # ---------------------------------------------------------

    baseline = df[df["policy_id"] == 0]
    heuristic = df[df["policy_id"] == 4]
    learned = df[df["policy_id"] == 99]

    if len(learned) > 0 and len(baseline) > 0:

        learned_correct = np.concatenate(learned["_per_problem_correct"].values)
        baseline_correct = np.concatenate(baseline["_per_problem_correct"].values)

        learned_vs_baseline = bootstrap_diff(
            learned_correct,
            baseline_correct
        )

        print("\nLearned vs Baseline:")
        print(learned_vs_baseline)

    if len(learned) > 0 and len(heuristic) > 0:

        heuristic_correct = np.concatenate(heuristic["_per_problem_correct"].values)

        learned_vs_heuristic = bootstrap_diff(
            learned_correct,
            heuristic_correct
        )

        print("\nLearned vs Heuristic:")
        print(learned_vs_heuristic)

    return df


# ==========================================================
# Main Entry
# ==========================================================

if __name__ == "__main__":

    # Policies to compare
    # 0 = baseline
    # 4 = heuristic energy
    # 99 = learned policy
    policies = [0, 4, 99]

    # Seeds for robustness
    seeds = (42, 43, 44)

    print("\n======================================")
    print(" Running Policy Comparison Experiment ")
    print("======================================")

    df = compare_policies(
        policy_ids=policies,
        seeds=seeds,
    )

    print("\n======================================")
    print(" Experiment Complete ")
    print("======================================")

    # Optional: save results
    output_path = "policy_experiment_results.csv"
    df.drop(columns=["_per_problem_correct"]).to_csv(output_path, index=False)

    print(f"\nResults saved to {output_path}")
