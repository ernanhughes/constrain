import pandas as pd
from constrain.runner import run
from constrain.analysis.stage2.application_evaluator import ApplicationEvaluator
from constrain.data.memory import Memory


import numpy as np


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

            memory = Memory()
            evaluator = ApplicationEvaluator(memory)

            summary, problem_df = evaluator.evaluate_run(run_id)

            summary["policy_id"] = pid
            summary["seed"] = seed

            # Store per-problem correctness for bootstrap
            summary["_per_problem_correct"] = problem_df["final_correct"].values

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
    energy = df[df["policy_id"] == 4]
    random = df[df["policy_id"] == 6]

    if len(energy) > 0 and len(baseline) > 0:

        energy_correct = np.concatenate(energy["_per_problem_correct"].values)
        baseline_correct = np.concatenate(baseline["_per_problem_correct"].values)

        energy_vs_baseline = bootstrap_diff(
            energy_correct,
            baseline_correct
        )

        print("\nEnergy vs Baseline:")
        print(energy_vs_baseline)

    if len(energy) > 0 and len(random) > 0:

        random_correct = np.concatenate(random["_per_problem_correct"].values)

        energy_vs_random = bootstrap_diff(
            energy_correct,
            random_correct
        )

        print("\nEnergy vs Random:")
        print(energy_vs_random)

    return df