#!/usr/bin/env python3

"""Long-horizon survival analysis for conversation stability.

This script is intentionally self-contained and uses only numpy/pandas/matplotlib.

Outputs:
  - survival_trials.csv
  - survival_statistics.csv
  - survival_curves.png
  - lifetime_pareto.png
  - intervention_rate_config.json (for matched-random baselines)
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from constrain.policy.eval.conversation_env import ConversationEnv
from constrain.policy.eval.intervention_rate_matcher import InterventionRateMatcher
from constrain.policy.eval.survival_prompts import get_stress_prompt
from constrain.policy.learned_policy import LearnedPolicy
from constrain.config import get_config


@dataclass
class SurvivalTrial:
    trial_id: int
    prompt_name: str
    controller: str
    failure_turn: int
    censored: bool
    max_turns: int
    interventions: int
    mean_V: float
    max_V: float
    cumulative_instability: float


class SurvivalAnalyzer:
    def __init__(self, config_path: str):
        with open(config_path, "r", encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f)
        self.app_cfg = get_config()

        self.trials: List[SurvivalTrial] = []
        self._matcher: InterventionRateMatcher | None = None

        # Learned policy for *action selection* (ACCEPT/REVERT) in policy-controlled arms.
        self._policy = None
        if getattr(self.app_cfg, "learned_model_path", None):
            self._policy = LearnedPolicy(
                model_path=self.app_cfg.learned_model_path,
                threshold=getattr(self.app_cfg, "learned_policy_threshold", 0.5),
            )

    # ------------------------------------------------------------
    # Failure criteria
    # ------------------------------------------------------------
    def check_failure(self, V_hist: List[float], acc_hist: List[float], energy_hist: List[float]) -> bool:
        crit = self.cfg["failure_criteria"]

        if crit["lyapunov_violation"]["enabled"]:
            tau = float(crit["lyapunov_violation"]["V_threshold"])
            k = int(crit["lyapunov_violation"]["consecutive_turns"])
            if len(V_hist) >= k and all(v > tau for v in V_hist[-k:]):
                return True

        if crit["accuracy_collapse"]["enabled"]:
            tau = float(crit["accuracy_collapse"]["threshold"])
            k = int(crit["accuracy_collapse"]["consecutive_turns"])
            if len(acc_hist) >= k and all(a < tau for a in acc_hist[-k:]):
                return True

        if crit["energy_spike"]["enabled"]:
            tau = float(crit["energy_spike"]["threshold"])
            if energy_hist and (energy_hist[-1] > tau):
                return True

        return False

    # ------------------------------------------------------------
    # Controller definitions
    # ------------------------------------------------------------
    def _choose_action(self, controller: str, *, feats: Dict[str, float], turn: int, trial_id: int) -> str:
        if controller == "baseline_no_control":
            return "ACCEPT"

        if controller == "truncate_random_matched":
            if self._matcher is None:
                raise RuntimeError("InterventionRateMatcher not initialized")
            return "REVERT" if self._matcher.should_intervene(turn=turn, seed=trial_id) else "ACCEPT"

        # Policy-driven controllers
        if self._policy is None:
            raise RuntimeError("learned_model_path not set; cannot run policy-controlled arms")

        # Explicit feature schema expected by LearnedPolicy training.
        feature_dict = {
            "iteration": float(turn),
            "temperature": float(feats.get("temperature", 0.0)),
            "ascii_ratio": float(feats.get("ascii_ratio", 1.0)),
            "foreign_char_ratio": float(feats.get("foreign_char_ratio", 0.0)),
            "repetition_score": float(feats.get("repetition_score", 0.0)),
            "total_energy": float(feats.get("total_energy", 0.0)),
            "grounding_energy": float(feats.get("grounding_energy", 0.0)),
            "stability_energy": float(feats.get("stability_energy", 0.0)),
        }

        action, _prob = self._policy.decide(feature_dict)
        return action

    # ------------------------------------------------------------
    # Trial execution
    # ------------------------------------------------------------
    def run_single_trial(self, *, trial_id: int, controller: str, prompt: Dict[str, str]) -> SurvivalTrial:
        max_turns = int(self.cfg["censoring"]["max_turns"])
        revert_cfg = self.cfg.get("revert", {})
        revert_mode = revert_cfg.get("mode", "hybrid")
        truncate_turns = int(revert_cfg.get("truncate_turns", 2))
        min_history_msgs = int(revert_cfg.get("min_history_msgs", 1))

        env = ConversationEnv(
            seed=trial_id,
            max_tokens=512,
            base_temperature=0.7,
            safe_temperature=0.0,
            learned_model_path=getattr(self.app_cfg, "learned_model_path", None),
        )
        state = env.reset(prompt["prompt"])

        V_hist: List[float] = []
        acc_hist: List[float] = []
        energy_hist: List[float] = []
        interventions = 0

        for turn in range(max_turns):
            # 1) Preview candidate
            response, raw, feats = env.preview_turn(state)

            # 2) Choose action
            action = self._choose_action(controller, feats=feats, turn=turn, trial_id=trial_id)

            # 3) Apply REVERT actuator (same turn), depending on arm.
            if action == "REVERT":
                interventions += 1

                if controller == "full_3head_temp_only":
                    prev = env.temperature
                    env.temperature = env.safe_temperature
                    response, raw, feats = env.preview_turn(state)
                    env.temperature = prev

                else:
                    # default: follow config revert.mode
                    if revert_mode in ("truncate_only", "hybrid"):
                        state = env.truncate_context(state, truncate_turns=truncate_turns, min_history_msgs=min_history_msgs)
                    if revert_mode in ("temp_only", "hybrid"):
                        prev = env.temperature
                        env.temperature = env.safe_temperature
                        response, raw, feats = env.preview_turn(state)
                        env.temperature = prev

            # 4) Commit exactly once
            state = env.commit_turn(state, response)

            # 5) Update histories
            V_hist.append(float(feats["collapse_prob"]))
            acc_hist.append(float(feats.get("accuracy", float("nan"))))
            energy_hist.append(float(feats.get("total_energy", 0.0)))

            # 6) Failure check
            if self.check_failure(V_hist, acc_hist, energy_hist):
                return SurvivalTrial(
                    trial_id=trial_id,
                    prompt_name=prompt["name"],
                    controller=controller,
                    failure_turn=turn,
                    censored=False,
                    max_turns=max_turns,
                    interventions=interventions,
                    mean_V=float(np.nanmean(V_hist)),
                    max_V=float(np.nanmax(V_hist)),
                    cumulative_instability=float(np.nansum([max(0.0, v - 0.5) for v in V_hist])),
                )

        # Censored
        return SurvivalTrial(
            trial_id=trial_id,
            prompt_name=prompt["name"],
            controller=controller,
            failure_turn=max_turns,
            censored=True,
            max_turns=max_turns,
            interventions=interventions,
            mean_V=float(np.nanmean(V_hist)) if V_hist else float("nan"),
            max_V=float(np.nanmax(V_hist)) if V_hist else float("nan"),
            cumulative_instability=float(np.nansum([max(0.0, v - 0.5) for v in V_hist])) if V_hist else 0.0,
        )

    # ------------------------------------------------------------
    # Kaplan–Meier + stats
    # ------------------------------------------------------------
    @staticmethod
    def kaplan_meier(times: np.ndarray, censored: np.ndarray, horizon: int) -> np.ndarray:
        # KM estimator on discrete turns 0..horizon
        n = len(times)
        surv = np.ones(horizon + 1, dtype=float)
        at_risk = n
        for t in range(horizon + 1):
            # events at t (not censored)
            events = np.sum((times == t) & (~censored))
            if at_risk > 0:
                surv[t] = (surv[t - 1] if t > 0 else 1.0) * (1.0 - (events / at_risk))
            else:
                surv[t] = surv[t - 1] if t > 0 else 0.0
            # remove all who end at t (events + censored)
            at_risk -= int(np.sum(times == t))
        return surv

    @staticmethod
    def rmst(surv: np.ndarray) -> float:
        # Restricted mean survival time is area under survival curve.
        return float(np.trapz(surv, dx=1.0))

    def compute_statistics(self, df: pd.DataFrame) -> pd.DataFrame:
        horizon = int(df["max_turns"].max())
        rows = []
        for ctrl, sub in df.groupby("controller"):
            times = sub["failure_turn"].to_numpy(dtype=int)
            cens = sub["censored"].to_numpy(dtype=bool)
            surv = self.kaplan_meier(times, cens, horizon=horizon)
            rows.append(
                {
                    "controller": ctrl,
                    "n_trials": int(len(sub)),
                    "rmst_0_to_T": self.rmst(surv),
                    "median_survival": int(np.argmax(surv <= 0.5)) if np.any(surv <= 0.5) else horizon,
                    "failure_rate": float((~cens).mean()),
                    "mean_interventions": float(sub["interventions"].mean()),
                }
            )
        stats = pd.DataFrame(rows)
        if "baseline_no_control" in set(stats["controller"]):
            base = float(stats.loc[stats["controller"] == "baseline_no_control", "rmst_0_to_T"].iloc[0])
            stats["rmst_ratio_vs_baseline"] = stats["rmst_0_to_T"] / base if base > 0 else np.nan
        return stats.sort_values("rmst_0_to_T", ascending=False).reset_index(drop=True)

    def plot_survival_curves(self, df: pd.DataFrame, out_path: Path) -> None:
        horizon = int(df["max_turns"].max())
        fig, ax = plt.subplots(figsize=(10, 6))
        for ctrl, sub in df.groupby("controller"):
            times = sub["failure_turn"].to_numpy(dtype=int)
            cens = sub["censored"].to_numpy(dtype=bool)
            surv = self.kaplan_meier(times, cens, horizon=horizon)
            ax.plot(np.arange(horizon + 1), surv, label=ctrl)
        ax.set_xlabel("Turn")
        ax.set_ylabel("P(Survival)")
        ax.set_ylim(0, 1.02)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="lower left")
        fig.tight_layout()
        fig.savefig(out_path, dpi=160)
        plt.close(fig)

    def plot_lifetime_pareto(self, stats: pd.DataFrame, out_path: Path) -> None:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(stats["mean_interventions"], stats["rmst_0_to_T"], s=200, alpha=0.7)
        for _, r in stats.iterrows():
            ax.annotate(r["controller"], (r["mean_interventions"], r["rmst_0_to_T"]), fontsize=9, ha="left")
        ax.set_xlabel("Mean interventions")
        ax.set_ylabel("RMST (0..T)")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_path, dpi=160)
        plt.close(fig)

    # ------------------------------------------------------------
    # Experiment orchestration
    # ------------------------------------------------------------
    def run(self) -> Tuple[pd.DataFrame, pd.DataFrame, Path]:
        exp = self.cfg.get("experiment", {})
        n = int(exp.get("n_trials_per_controller", 30))
        controllers = list(exp.get("controllers", ["baseline_no_control", "full_3head"]))
        seed = int(exp.get("seed", 42))

        # Prepare output
        base_out = Path(self.cfg.get("reporting", {}).get("output_dir", "outputs/survival_analysis"))
        out_dir = base_out / datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir.mkdir(parents=True, exist_ok=True)

        # 1) If we have a matched-random arm, we need a reference intervention rate.
        # We compute it from the *policy-controlled hybrid* arm by default.
        need_match = "truncate_random_matched" in controllers
        rate_path = out_dir / "intervention_rate_config.json"

        # Run controllers in deterministic order
        rng = np.random.default_rng(seed)
        trial_id = 0
        reference_trials: List[Dict] = []

        # First pass: run full controller (if present) so we can compute rate
        if need_match and "full_3head" in controllers:
            for i in range(n):
                prompt = get_stress_prompt(seed=seed + i)
                t = self.run_single_trial(trial_id=trial_id, controller="full_3head", prompt=prompt)
                self.trials.append(t)
                reference_trials.append(asdict(t))
                trial_id += 1
            self._matcher = InterventionRateMatcher.from_reference_trials(reference_trials)
            self._matcher.save(rate_path)

        # Second pass: run remaining controllers
        for ctrl in controllers:
            if ctrl == "full_3head" and need_match:
                # already ran
                continue
            if ctrl == "truncate_random_matched":
                # load matcher
                if self._matcher is None and rate_path.exists():
                    self._matcher = InterventionRateMatcher.load(rate_path)
                if self._matcher is None:
                    raise RuntimeError("Missing intervention rate config; run full_3head first")

            for i in range(n):
                prompt = get_stress_prompt(seed=seed + 10_000 + i)
                t = self.run_single_trial(trial_id=trial_id, controller=ctrl, prompt=prompt)
                self.trials.append(t)
                trial_id += 1

        df = pd.DataFrame([asdict(t) for t in self.trials])
        stats = self.compute_statistics(df)

        # Save
        df.to_csv(out_dir / "survival_trials.csv", index=False)
        stats.to_csv(out_dir / "survival_statistics.csv", index=False)
        self.plot_survival_curves(df, out_dir / "survival_curves.png")
        self.plot_lifetime_pareto(stats, out_dir / "lifetime_pareto.png")

        return df, stats, out_dir


def main() -> None:
    config_path = "constrain/policy/eval/survival_config.yaml"
    analyzer = SurvivalAnalyzer(config_path)
    df, stats, out_dir = analyzer.run()

    print("\n=== Survival statistics ===")
    print(stats.to_string(index=False))
    print(f"\nOutputs: {out_dir}")


if __name__ == "__main__":
    main()
