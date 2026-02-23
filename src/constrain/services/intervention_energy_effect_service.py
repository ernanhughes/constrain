"""constrain.services.intervention_energy_effect_service
Energy Delta Diagnostic Service

Measures whether intervention reduces short-term instability.
Instead of collapse_{t+1}, measures:
    ΔE = E_{t+1} - E_t

Provides mechanistic evidence, not just outcome evidence.
"""

from typing import Dict, Tuple
import numpy as np
import pandas as pd
from constrain.data.memory import Memory
from constrain.config import get_config


class InterventionEnergyEffectService:
    def __init__(self, memory: Memory, n_bootstrap: int = 2000, seed: int = 42):
        self.memory = memory
        self.n_bootstrap = n_bootstrap
        self.seed = seed
        self.cfg = get_config()

    def analyze_run(self, run_id: str) -> Dict:
        """
        Analyze energy change after intervention vs accept.

        Returns:
            {
                "mean_delta_accept": float,
                "mean_delta_intervene": float,
                "difference": float,
                "bootstrap_ci": (low, high),
                "p_value": float,
                "n_accept": int,
                "n_intervene": int,
                "effect_size": float,
                "verdict": str
            }
        """
        # Load steps
        steps = self.memory.steps.get_by_run(run_id)
        if not steps:
            return self._empty_result("no_steps")

        df = pd.DataFrame([s.model_dump() for s in steps])
        df = df.sort_values(["problem_id", "iteration"]).reset_index(drop=True)

        # Compute energy delta per step
        df["energy_delta"] = df.groupby("problem_id")["total_energy"].diff()
        df = df.dropna(subset=["energy_delta"])

        # Separate by action
        accept_mask = df["policy_action"] == "ACCEPT"
        intervene_mask = df["policy_action"] != "ACCEPT"

        delta_accept = df.loc[accept_mask, "energy_delta"].values
        delta_intervene = df.loc[intervene_mask, "energy_delta"].values

        if len(delta_accept) < 10 or len(delta_intervene) < 10:
            return self._empty_result("insufficient_samples")

        # Compute means
        mean_accept = float(np.mean(delta_accept))
        mean_intervene = float(np.mean(delta_intervene))
        difference = mean_intervene - mean_accept

        # Bootstrap CI
        ci_low, ci_high = self._bootstrap_ci(delta_accept, delta_intervene)

        # P-value via permutation test
        p_value = self._permutation_test(delta_accept, delta_intervene)

        # Effect size (Cohen's d)
        pooled_std = np.sqrt(
            (np.var(delta_accept) + np.var(delta_intervene)) / 2
        )
        effect_size = difference / pooled_std if pooled_std > 0 else 0

        result = {
            "mean_delta_accept": mean_accept,
            "mean_delta_intervene": mean_intervene,
            "difference": difference,
            "bootstrap_ci": (ci_low, ci_high),
            "p_value": p_value,
            "n_accept": len(delta_accept),
            "n_intervene": len(delta_intervene),
            "effect_size": effect_size,
        }

        result["verdict"] = self._compute_verdict(result)

        return result

    def _bootstrap_ci(
        self, delta_accept: np.ndarray, delta_intervene: np.ndarray
    ) -> Tuple[float, float]:
        rng = np.random.RandomState(self.seed)
        diffs = []

        for _ in range(self.n_bootstrap):
            idx_a = rng.choice(len(delta_accept), len(delta_accept), replace=True)
            idx_i = rng.choice(len(delta_intervene), len(delta_intervene), replace=True)

            diff = delta_intervene[idx_i].mean() - delta_accept[idx_a].mean()
            diffs.append(diff)

        return float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))

    def _permutation_test(
        self, delta_accept: np.ndarray, delta_intervene: np.ndarray, n_perm: int = 5000
    ) -> float:
        rng = np.random.RandomState(self.seed)
        observed = delta_intervene.mean() - delta_accept.mean()

        combined = np.concatenate([delta_accept, delta_intervene])
        n_a = len(delta_accept)

        extreme_count = 0
        for _ in range(n_perm):
            rng.shuffle(combined)
            perm_diff = combined[n_a:].mean() - combined[:n_a].mean()
            if abs(perm_diff) >= abs(observed):
                extreme_count += 1

        return extreme_count / n_perm

    def _compute_verdict(self, stats: Dict) -> str:
        diff = stats["difference"]
        ci = stats["bootstrap_ci"]
        p = stats["p_value"]

        if diff > 0.02 and ci[0] > 0:
            return "⚠️  INTERVENTION INCREASES ENERGY (harmful)"
        elif diff < -0.02 and ci[1] < 0:
            return "✅ INTERVENTION REDUCES ENERGY (beneficial)"
        elif p < 0.05:
            return "⚠️  SIGNIFICANT EFFECT (check direction)"
        else:
            return "⚪ NO SIGNIFICANT ENERGY EFFECT"

    def _empty_result(self, reason: str) -> Dict:
        return {
            "mean_delta_accept": None,
            "mean_delta_intervene": None,
            "difference": None,
            "bootstrap_ci": (None, None),
            "p_value": None,
            "n_accept": 0,
            "n_intervene": 0,
            "effect_size": None,
            "verdict": f"SKIPPED: {reason}",
        }


def main():
    from constrain.config import get_config
    from constrain.data.memory import Memory

    cfg = get_config()
    memory = Memory(cfg.db_url)

    runs = memory.runs.get_recent(limit=1)
    if not runs:
        print("❌ No runs found")
        return

    run_id = runs[0].run_id
    print(f"Analyzing run: {run_id}")

    service = InterventionEnergyEffectService(memory)
    result = service.analyze_run(run_id)

    print("\n" + "=" * 60)
    print("ENERGY DELTA DIAGNOSTIC")
    print("=" * 60)
    print(f"Verdict: {result['verdict']}")
    print("\nSample sizes:")
    print(f"  Accept transitions: {result['n_accept']}")
    print(f"  Intervene transitions: {result['n_intervene']}")
    print("\nEnergy change (ΔE = E_t+1 - E_t):")
    print(f"  Mean ΔE (accept): {result['mean_delta_accept']:.4f}")
    print(f"  Mean ΔE (intervene): {result['mean_delta_intervene']:.4f}")
    print(f"  Difference: {result['difference']:.4f}")
    print("\nStatistical tests:")
    print(f"  95% CI: [{result['bootstrap_ci'][0]:.4f}, {result['bootstrap_ci'][1]:.4f}]")
    print(f"  P-value: {result['p_value']:.4f}")
    print(f"  Effect size (Cohen's d): {result['effect_size']:.3f}")
    print("=" * 60)


if __name__ == "__main__":
    main()