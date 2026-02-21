import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from constrain.analysis.stage2.application_evaluator import \
    ApplicationEvaluator
from constrain.config import get_config
from constrain.data.memory import Memory
from constrain.runner import run

THRESHOLDS = np.arange(0.05, 0.51, 0.05)
SEED = 42
NUM_PROBLEMS = 20


def run_single_threshold(threshold):

    print(f"\n🚀 Running threshold={threshold:.2f}")

    run_id = run(
        policy_id=99,
        seed=SEED,
        threshold=threshold,
        num_problems=NUM_PROBLEMS,
    )

    memory = Memory()
    evaluator = ApplicationEvaluator(memory)
    summary, problem_df = evaluator.evaluate_run(run_id)

    return {
        "threshold": threshold,
        "accuracy": summary["accuracy"],
        "collapse_rate": problem_df["collapsed"].mean(),
        "intervention_rate": summary["intervention_rate"],
    }


def main():

    results = []

    for t in THRESHOLDS:
        row = run_single_threshold(t)
        results.append(row)

    df = pd.DataFrame(results)
    get_config().output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv("threshold_sweep_results.csv", index=False)

    print("\n==============================")
    print(df)
    print("==============================")

    plot_results(df)
    compute_pareto(df)

def plot_results(df):

    plt.figure(figsize=(10, 6))

    plt.plot(df["threshold"], df["collapse_rate"], label="Collapse Rate")
    plt.plot(df["threshold"], df["accuracy"], label="Accuracy")
    plt.plot(df["threshold"], df["intervention_rate"], label="Intervention Rate")

    plt.xlabel("Threshold")
    plt.ylabel("Metric Value")
    plt.legend()
    plt.title("Threshold Sweep Analysis")

    plt.tight_layout()
    plt.savefig("threshold_sweep_plot.png")
    plt.show()


def compute_pareto(df):

    pareto = []

    for i, row in df.iterrows():
        dominated = False
        for j, other in df.iterrows():
            if (
                other["accuracy"] >= row["accuracy"]
                and other["collapse_rate"] <= row["collapse_rate"]
                and (
                    other["accuracy"] > row["accuracy"]
                    or other["collapse_rate"] < row["collapse_rate"]
                )
            ):
                dominated = True
                break

        if not dominated:
            pareto.append(row)

    pareto_df = pd.DataFrame(pareto)
    print("\nPareto Frontier:")
    print(pareto_df)

    return pareto_df

if __name__ == "__main__":
    main()