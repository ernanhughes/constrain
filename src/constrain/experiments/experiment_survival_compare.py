from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import asdict
from typing import Dict, Tuple

from constrain.policy.eval.run_survival_analysis import SurvivalAnalyzer
from constrain.policy.eval.survival_prompts import get_stress_prompt
from constrain.config import get_config

# ============================================================
# Bootstrap helpers (similar to your style)
# ============================================================

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


def rmst(times: np.ndarray, censored: np.ndarray, T: int) -> float:
    """
    Restricted mean survival time RMST(0..T).
    Discrete-time Kaplan-Meier style integration.

    times: failure_turn (or censoring turn)
    censored: True if right-censored
    """
    # Guard
    if len(times) == 0:
        return float("nan")

    # KM: S(t) piecewise at each integer turn
    # We compute survival after each turn t (0..T-1)
    at_risk = len(times)
    S = 1.0
    area = 0.0

    # For each turn t: area += S(t)
    # event: time == t AND not censored
    # remove: time == t (censored or not) leaves risk set after t
    for t in range(T):
        area += S

        events = int(np.sum((times == t) & (~censored)))
        if at_risk > 0:
            hazard = events / at_risk
            S *= (1.0 - hazard)

        removed = int(np.sum(times == t))
        at_risk -= removed
        if at_risk <= 0:
            break

    return float(area)


def median_survival(times: np.ndarray, censored: np.ndarray, T: int) -> float:
    """
    Median survival time from KM curve: first t where S(t) <= 0.5.
    Returns NaN if never crosses within [0, T].
    """
    if len(times) == 0:
        return float("nan")

    at_risk = len(times)
    S = 1.0

    for t in range(T):
        events = int(np.sum((times == t) & (~censored)))
        if at_risk > 0:
            hazard = events / at_risk
            S *= (1.0 - hazard)

        removed = int(np.sum(times == t))
        at_risk -= removed

        if S <= 0.5:
            return float(t)

        if at_risk <= 0:
            break

    return float("nan")


# ============================================================
# Core experiment
# ============================================================

def run_survival_controller_trials(
    *,
    analyzer: SurvivalAnalyzer,
    controller: str,
    seed: int,
    n_trials: int,
    max_turns: int,
    prompt_category: str = "all",
) -> pd.DataFrame:
    """
    Runs n_trials single-trial survival runs for a controller at a fixed seed.
    Returns per-trial dataframe.
    """
    rows = []
    for i in range(n_trials):
        prompt = get_stress_prompt(category=prompt_category, seed=seed + i)

        try:
            trial = analyzer.run_single_trial(
                trial_id=(seed * 100000 + i),   # stable unique id
                prompt=prompt,
                controller=controller,
                max_turns=max_turns,
            )
        except Exception as e:
            # Robust: record failure to run as NaN row
            rows.append({
                "controller": controller,
                "seed": seed,
                "trial_index": i,
                "prompt_name": prompt.get("name", "unknown"),
                "error": str(e),
                "failure_turn": np.nan,
                "censored": np.nan,
                "interventions": np.nan,
                "mean_V": np.nan,
                "max_V": np.nan,
                "cumulative_instability": np.nan,
            })
            continue

        d = asdict(trial)
        rows.append({
            "controller": controller,
            "seed": seed,
            "trial_index": i,
            "prompt_name": d.get("prompt_name"),
            "error": "",
            "failure_turn": d.get("failure_turn"),
            "censored": d.get("censored"),
            "interventions": d.get("interventions"),
            "mean_V": d.get("mean_V"),
            "max_V": d.get("max_V"),
            "cumulative_instability": d.get("cumulative_instability"),
        })

    return pd.DataFrame(rows)


def summarize_controller(
    trials_df: pd.DataFrame,
    controller: str,
    max_turns: int,
) -> Dict:
    sub = trials_df[(trials_df["controller"] == controller) & (trials_df["error"] == "")]
    times = sub["failure_turn"].to_numpy(dtype=int) if len(sub) else np.array([], dtype=int)
    cens = sub["censored"].to_numpy(dtype=bool) if len(sub) else np.array([], dtype=bool)

    # If we have no valid trials:
    if len(sub) == 0:
        return {
            "controller": controller,
            "n_trials": 0,
            "rmst_0_T": np.nan,
            "median_survival": np.nan,
            "failure_rate": np.nan,
            "mean_interventions": np.nan,
            "mean_V": np.nan,
            "mean_max_V": np.nan,
        }

    rmst_val = rmst(times, cens, T=max_turns)
    med_val = median_survival(times, cens, T=max_turns)

    failure_rate = float(np.mean(~cens))  # proportion uncensored = observed failure
    mean_interventions = float(sub["interventions"].mean())
    mean_V = float(sub["mean_V"].mean())
    mean_max_V = float(sub["max_V"].mean())

    return {
        "controller": controller,
        "n_trials": int(len(sub)),
        "rmst_0_T": rmst_val,
        "median_survival": med_val,
        "failure_rate": failure_rate,
        "mean_interventions": mean_interventions,
        "mean_V": mean_V,
        "mean_max_V": mean_max_V,
    }


def compare_controllers(
    *,
    config_path: str = "src/constrain/policy/eval/survival_config.yaml",
    model_base_path: str = get_config().learned_model_path,
    controllers: Tuple[str, ...] = (
        "baseline_no_control",
        "full_3head_temp_only",
        "full_3head_hybrid",
        "truncate_random_matched",
    ),
    seeds: Tuple[int, ...] = (42, 43, 44),
    n_trials_per_seed: int = 10,
    max_turns: int = 300,
    prompt_category: str = "all",
    output_csv: str = f"{get_config().reports_dir}/survival_controller_experiment.csv",
    bootstrap_n: int = 2000,
):
    """
    End-to-end experiment:
      - runs trials for each controller across seeds
      - summarizes survival metrics
      - bootstraps diffs vs baseline
    """
    analyzer = SurvivalAnalyzer(config_path=config_path, model_base_path=model_base_path)

    all_trials = []
    for seed in seeds:
        print("\n==============================")
        print(f"Seed {seed}")
        print("==============================")
        for ctrl in controllers:
            print(f"▶ Running controller={ctrl}  trials={n_trials_per_seed}")
            df_trials = run_survival_controller_trials(
                analyzer=analyzer,
                controller=ctrl,
                seed=seed,
                n_trials=n_trials_per_seed,
                max_turns=max_turns,
                prompt_category=prompt_category,
            )
            all_trials.append(df_trials)

    trials_df = pd.concat(all_trials, ignore_index=True)
    trials_df.to_csv(output_csv, index=False)
    print(f"\n📄 Saved per-trial results to: {output_csv}")

    # Summaries
    summaries = []
    for ctrl in controllers:
        summaries.append(summarize_controller(trials_df, ctrl, max_turns=max_turns))
    summary_df = pd.DataFrame(summaries)

    print("\n==============================")
    print("Survival Summary")
    print("==============================")
    print(summary_df.to_string(index=False))

    # Bootstrap diffs vs baseline on per-trial lifetimes (uncensored treated as observed, censored at max_turns)
    baseline = trials_df[(trials_df["controller"] == controllers[0]) & (trials_df["error"] == "")]
    baseline_times = baseline["failure_turn"].to_numpy(dtype=float)

    print("\n==============================")
    print("Bootstrap diffs vs Baseline")
    print("==============================")
    for ctrl in controllers[1:]:
        sub = trials_df[(trials_df["controller"] == ctrl) & (trials_df["error"] == "")]
        cur_times = sub["failure_turn"].to_numpy(dtype=float)

        stats = bootstrap_diff(cur_times, baseline_times, n=bootstrap_n, seed=42)
        print(f"\n{ctrl} vs {controllers[0]} (failure_turn):")
        print(stats)

    return trials_df, summary_df


if __name__ == "__main__":
    print("\n======================================")
    print(" Long-Conversation Controller Experiment ")
    print("======================================")

    # Small demo defaults:
    #  - 3 seeds
    #  - 10 trials/seed/controller = 30 trials/controller
    # You can change without touching YAML.
    model_path  = get_config().learned_model_path
    trials_df, summary_df = compare_controllers(
        n_trials_per_seed=10,
        max_turns=200,  # shorten for quick test
        model_base_path=model_path,
    )

    print("\n======================================")
    print(" Experiment Complete ")
    print("======================================")

    # Optional: also save summary
    if not summary_df.empty:
        out = f"{get_config().reports_dir}/survival_controller_summary.csv"
        summary_df.to_csv(out, index=False)
        print(f"📄 Saved summary to {out}")