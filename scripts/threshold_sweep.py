# scripts/threshold_sweep.py (corrected version)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

from constrain.config import get_config
from constrain.data.memory import Memory
from constrain.services.policy_evaluation_service import PolicyEvaluationService
from constrain.policy.threshold.thresholds import FixedThresholdProvider
from constrain.core.runner import run_with_provider  # ← Use new function

# ============================================================
# Configuration
# ============================================================
THRESHOLDS = np.arange(0.05, 0.51, 0.05)
SEED = 42
NUM_PROBLEMS = 20
POLICY_ID = 99

TAU_MEDIUM_RATIO = 1.5
TAU_HARD_RATIO = 2.0


def run_single_threshold(threshold: float, policy_id: int = POLICY_ID) -> dict:
    """Run a single threshold value using FixedThresholdProvider."""
    
    # Create provider with fixed thresholds (no config modification needed)
    provider = FixedThresholdProvider(
        tau_soft=threshold,
        tau_medium=threshold * TAU_MEDIUM_RATIO,
        tau_hard=threshold * TAU_HARD_RATIO,
    )
    
    # Run with injected provider
    run_id = run_with_provider(
        policy_id=policy_id,
        threshold_provider=provider,  # ← Inject directly
        seed=SEED,
        num_problems=NUM_PROBLEMS,
    )
    
    if run_id is None:
        raise ValueError("run_with_provider() returned None")
    
    # Evaluate using consolidated service
    cfg = get_config()
    memory = Memory(cfg.db_url)
    evaluator = PolicyEvaluationService(memory)
    summary, problem_df = evaluator.evaluate_run(run_id)
    
    return {
        "threshold": threshold,
        "tau_soft": threshold,
        "tau_medium": threshold * TAU_MEDIUM_RATIO,
        "tau_hard": threshold * TAU_HARD_RATIO,
        "accuracy": summary.accuracy,
        "collapse_rate": float(problem_df["collapsed"].mean()) if "collapsed" in problem_df.columns else 0.0,
        "intervention_rate": summary.intervention_rate,
        "avg_energy": summary.avg_energy,
        "run_id": run_id,
    }


def main():
    results = []
    
    print(f"\n🔍 Starting threshold sweep: {THRESHOLDS[0]:.2f} → {THRESHOLDS[-1]:.2f}")
    
    for t in THRESHOLDS:
        try:
            row = run_single_threshold(t)
            results.append(row)
            print(f"  ✓ threshold={t:.2f}: acc={row['accuracy']:.2%}, collapse={row['collapse_rate']:.2%}")
        except Exception as e:
            print(f"  ✗ threshold={t:.2f}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    if not results:
        print("❌ No successful runs")
        return
    
    df = pd.DataFrame(results)
    
    # Save results
    output_dir = Path(get_config().reports_dir) if hasattr(get_config(), "reports_dir") else Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    df.to_csv(output_dir / "threshold_sweep_results.csv", index=False)
    print(f"\n✅ Results saved to {output_dir / 'threshold_sweep_results.csv'}")
    
    # Display summary
    print("\n" + "="*60)
    print(df[["threshold", "accuracy", "collapse_rate", "intervention_rate"]])
    print("="*60)
    
    # Visualize
    plot_results(df, output_dir)
    
    # Compute Pareto frontier
    pareto = compute_pareto(df)
    if not pareto.empty:
        print(f"\n🎯 Pareto-optimal thresholds: {pareto['threshold'].tolist()}")


def plot_results(df: pd.DataFrame, output_dir: Path):
    """Generate comparison plots."""
    plt.figure(figsize=(12, 5))
    
    # Accuracy + Collapse Rate
    plt.subplot(1, 2, 1)
    plt.plot(df["threshold"], df["accuracy"] * 100, label="Accuracy", marker="o")
    plt.plot(df["threshold"], df["collapse_rate"] * 100, label="Collapse Rate", marker="s")
    plt.xlabel("Tau Soft Threshold")
    plt.ylabel("Percentage (%)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.title("Accuracy vs. Collapse Rate")
    
    # Intervention Rate
    plt.subplot(1, 2, 2)
    plt.plot(df["threshold"], df["intervention_rate"] * 100, label="Intervention Rate", marker="^", color="orange")
    plt.xlabel("Tau Soft Threshold")
    plt.ylabel("Percentage (%)")
    plt.grid(alpha=0.3)
    plt.title("Intervention Frequency")
    
    plt.tight_layout()
    plt.savefig(output_dir / "threshold_sweep_plot.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"📊 Plot saved to {output_dir / 'threshold_sweep_plot.png'}")


def compute_pareto(df: pd.DataFrame) -> pd.DataFrame:
    """Compute Pareto frontier."""
    pareto = []
    
    for i, row in df.iterrows():
        dominated = False
        for j, other in df.iterrows():
            if (other["accuracy"] >= row["accuracy"] and 
                other["collapse_rate"] <= row["collapse_rate"] and
                (other["accuracy"] > row["accuracy"] or 
                 other["collapse_rate"] < row["collapse_rate"])):
                dominated = True
                break
        if not dominated:
            pareto.append(row)
    
    return pd.DataFrame(pareto).sort_values("threshold").reset_index(drop=True)


if __name__ == "__main__":
    main()