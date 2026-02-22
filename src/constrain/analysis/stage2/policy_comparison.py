import numpy as np
import pandas as pd

from constrain.analysis.stage2.application_evaluator import ApplicationEvaluator
from constrain.config import get_config
from constrain.data.memory import Memory
from constrain.runner import run


def bootstrap_diff(a, b, n=2000, seed=42):
    rng = np.random.RandomState(seed)
    diffs = []

    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)

    if len(a) == 0 or len(b) == 0:
        return {"mean_diff": np.nan, "ci_lower": np.nan, "ci_upper": np.nan}

    for _ in range(n):
        idx_a = rng.choice(len(a), len(a), replace=True)
        idx_b = rng.choice(len(b), len(b), replace=True)
        diffs.append(a[idx_a].mean() - b[idx_b].mean())

    lower = np.percentile(diffs, 2.5)
    upper = np.percentile(diffs, 97.5)

    return {
        "mean_diff": float(np.mean(diffs)),
        "ci_lower": float(lower),
        "ci_upper": float(upper),
    }


def _collapsed_flags_from_steps(steps_df: pd.DataFrame) -> pd.Series:
    """
    Returns a boolean Series indexed by problem_id indicating if that problem "collapsed".

    Priority order:
      1) collapse_probability column (>=0.5 considered collapsed)
      2) phase column containing 'collapse'
      3) fallback: all False
    """
    if steps_df.empty:
        return pd.Series(dtype=bool)

    if "collapse_probability" in steps_df.columns and steps_df["collapse_probability"].notna().any():
        return (
            steps_df.groupby("problem_id")["collapse_probability"]
            .apply(lambda x: (x.fillna(0.0) >= 0.5).any())
        )

    if "phase" in steps_df.columns:
        return (
            steps_df.groupby("problem_id")["phase"]
            .apply(lambda x: (x.astype(str) == "collapse").any())
        )

    return pd.Series(False, index=steps_df["problem_id"].unique())


def compare_policies(policy_ids, seeds=(42, 43, 44), num_problems: int | None = None):
    cfg = get_config()
    memory = Memory(cfg.db_url)
    evaluator = ApplicationEvaluator(memory)

    all_rows = []

    for seed in seeds:
        print("\n==============================")
        print(f"Running seed {seed}")
        print("==============================")

        for pid in policy_ids:
            print(f"\n🚀 Policy {pid} (seed={seed})")

            # ---------------------------------------------------------
            # Run experiment
            # ---------------------------------------------------------
            try:
                run_id = run(policy_id=pid, seed=seed, num_problems=num_problems)
            except Exception as e:
                print(f"⚠ Run failed for policy {pid}, seed {seed}: {e}")
                continue

            # ---------------------------------------------------------
            # Evaluate run (robust)
            # ---------------------------------------------------------
            try:
                summary, problem_df = evaluator.evaluate_run(run_id)
            except Exception as e:
                print(f"⚠ Evaluation failed for run {run_id}: {e}")
                continue

            summary["policy_id"] = pid
            summary["seed"] = seed
            summary["run_id"] = run_id

            # Per-problem correctness for bootstrap
            if "final_correct" in problem_df.columns:
                summary["_per_problem_correct"] = problem_df["final_correct"].values.astype(float)
            else:
                summary["_per_problem_correct"] = np.array([], dtype=float)

            # ---------------------------------------------------------
            # Collapse per-problem (from steps)
            # ---------------------------------------------------------
            try:
                steps = memory.steps.get_by_run(run_id)
                steps_df = pd.DataFrame([s.model_dump() for s in steps])
            except Exception as e:
                print(f"⚠ Could not load steps for run {run_id}: {e}")
                steps_df = pd.DataFrame()

            collapsed_per_problem = _collapsed_flags_from_steps(steps_df)

            if not collapsed_per_problem.empty and "problem_id" in problem_df.columns:
                problem_df["collapsed"] = problem_df["problem_id"].map(collapsed_per_problem).fillna(False)
                summary["collapse_rate"] = float(problem_df["collapsed"].mean())
            else:
                summary["collapse_rate"] = float("nan")

            # Store
            all_rows.append(summary)

    if len(all_rows) == 0:
        print("❌ No successful runs.")
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)

    print("\n==============================")
    print("Raw Results")
    print("==============================")
    display_cols = [c for c in df.columns if c != "_per_problem_correct"]
    print(df[display_cols])

    # ---------------------------------------------------------
    # Statistical Comparisons (bootstrap)
    # ---------------------------------------------------------
    def _collect(policy_id: int) -> np.ndarray:
        sub = df[df["policy_id"] == policy_id]
        if len(sub) == 0:
            return np.array([], dtype=float)
        return np.concatenate(sub["_per_problem_correct"].values)

    baseline_correct = _collect(0)

    # Compare each non-baseline to baseline
    for pid in policy_ids:
        if pid == 0:
            continue
        cur = _collect(pid)
        if len(cur) == 0 or len(baseline_correct) == 0:
            print(f"\n(pid={pid}) Skipping bootstrap (missing data).")
            continue

        stats = bootstrap_diff(cur, baseline_correct)
        print(f"\nPolicy {pid} vs Baseline (0):")
        print(stats)

    return df


if __name__ == "__main__":
    policies = [0, 4, 99]  # baseline, heuristic, learned
    seeds = (42, 43, 44)

    # You can tune this without touching TOML
    # (use None to fall back to cfg.num_problems)
    num_problems = 20

    print("\n======================================")
    print(" Running Policy Comparison Experiment ")
    print("======================================")

    df = compare_policies(
        policy_ids=policies,
        seeds=seeds,
        num_problems=num_problems,
    )

    print("\n======================================")
    print(" Experiment Complete ")
    print("======================================")

    if not df.empty:
        output_path = "policy_experiment_results.csv"
        df.drop(columns=["_per_problem_correct"], errors="ignore").to_csv(output_path, index=False)
        print(f"\nResults saved to {output_path}")