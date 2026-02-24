from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

from constrain.control.rl_parameter_adapter import (ActionBounds,
                                                    RLParameterAdapter)
from constrain.core.io.artifact_store import ArtifactStore
from constrain.core.io.markdown_report import MarkdownReport
from constrain.core.metrics.safety_tax import (compute_safety_tax,
                                               format_safety_tax)
from constrain.experiments.base import Experiment, ExperimentResult
from constrain.experiments.eval_rl_adapter import (EvalConfig, RunMetrics,
                                                   eval_one)


@dataclass
class Paper3EvalConfig:
    checkpoint: str = ""  # optional for the lightweight adapter; used by your full adapter
    steps: int = 2000
    batch_size: int = 32
    seeds: Tuple[int, ...] = (0, 1, 2)
    block_sizes: Tuple[int, ...] = (32, 16)
    mixture_alphas: Tuple[float, ...] = (0.1, 0.5, 5.0)
    static_tau: float = 8.0
    static_temp: float = 1.0
    static_rec: int = 4
    leakage: float = 0.9


class Paper3SRGEvalExperiment(Experiment):
    name = "paper3_srg_eval"

    def run(self, out_root: Path, cfg: Paper3EvalConfig = Paper3EvalConfig()) -> ExperimentResult:
        run_dir = self._make_run_dir(out_root, f"{self.name}_{int(time.time())}")
        store = ArtifactStore(run_dir)

        bounds = ActionBounds()
        adapter = RLParameterAdapter(bounds=bounds, checkpoint_path=(cfg.checkpoint or None))

        eval_cfg = EvalConfig(
            checkpoint=cfg.checkpoint or "",
            steps=cfg.steps,
            batch_size=cfg.batch_size,
            seeds=list(cfg.seeds),
            leakage=cfg.leakage,
            block_sizes=tuple(cfg.block_sizes),
            mixture_alphas=tuple(cfg.mixture_alphas),
            static_tau=cfg.static_tau,
            static_temp=cfg.static_temp,
            static_rec=cfg.static_rec,
        )

        env_specs: List[Tuple[str, str, int, float]] = []
        for bs in cfg.block_sizes:
            env_specs.append((f"blocks_{bs}", "blocks", int(bs), 0.2))
        for a in cfg.mixture_alphas:
            env_specs.append((f"mixture_{a}", "mixture", 32, float(a)))

        all_runs: List[Dict[str, Any]] = []
        t0 = time.time()

        for env_name, sched_type, bs, alpha in env_specs:
            for seed in cfg.seeds:
                for model_id in ["S", "R", "G"]:
                    rm: RunMetrics = eval_one(
                        cfg=eval_cfg,
                        adapter=adapter,
                        model_id=model_id,
                        schedule_type=sched_type,
                        block_size=bs,
                        alpha=alpha,
                        seed=int(seed),
                        steps_csv_writer=None,
                    )
                    all_runs.append(asdict(rm))

        elapsed = time.time() - t0

        def mean(xs): return sum(xs) / max(1, len(xs))

        envs = sorted({r["env"] for r in all_runs})
        summary: Dict[str, Dict[str, Any]] = {}

        for env in envs:
            summary[env] = {}
            for model_id in ["S", "R", "G"]:
                rs = [r for r in all_runs if r["env"] == env and r["model"] == model_id]
                if not rs:
                    continue
                summary[env][model_id] = {
                    "collapse_rate": mean([r["collapse_rate"] for r in rs]),
                    "perf": mean([r["performance_proxy"] for r in rs]),
                    "fallback_rate": mean([r["fallback_rate"] for r in rs]),
                    "jitter": mean([r["avg_jitter"] for r in rs]),
                }

            if "S" in summary[env] and "G" in summary[env]:
                s = summary[env]["S"]
                g = summary[env]["G"]
                st = compute_safety_tax(g["perf"], s["perf"], g["collapse_rate"], s["collapse_rate"])
                summary[env]["SafetyTax_G_vs_S"] = {
                    "value": st.safety_tax,
                    "formatted": format_safety_tax(st),
                    "collapse_reduction": st.collapse_reduction,
                    "perf_delta": st.perf_delta,
                }

        cfg_path = store.write_json("config.json", asdict(cfg))
        runs_csv = store.write_csv("runs.csv", all_runs)
        summary_path = store.write_json("summary.json", {"elapsed_sec": elapsed, "by_env": summary})

        rpt = MarkdownReport("Paper 3 — S/R/G Evaluation (Adaptive Governance)")
        rpt.add_meta(experiment=self.name, elapsed_sec=elapsed, run_dir=str(run_dir))

        rpt.add_section(
            "What this experiment tests",
            "\n".join([
                "**S/R/G framing:**",
                "- **S** = Static best (fixed params)",
                "- **R** = RL adapter ungated (clamp only)",
                "- **G** = RL adapter safety-gated (clamp→Lyapunov fallback)",
                "",
                "**Claim:** Safety-gated adaptation reduces collapse under shift with bounded cost (SafetyTax < 1.0).",
            ])
        )

        table = ["| Env | Collapse S↓ | Collapse R↓ | Collapse G↓ | SafetyTax (G vs S) |",
                 "|---|---:|---:|---:|---|"]
        for env in envs:
            s = summary[env].get("S", {})
            r = summary[env].get("R", {})
            g = summary[env].get("G", {})
            st = summary[env].get("SafetyTax_G_vs_S", {})
            table.append(
                f"| {env} | {s.get('collapse_rate', 0.0):.4f} | {r.get('collapse_rate', 0.0):.4f} | {g.get('collapse_rate', 0.0):.4f} | {st.get('formatted','N/A')} |"
            )

        rpt.add_section("Results (mean over seeds)", "\n".join(table))

        rpt.add_section(
            "Interpretation",
            "\n".join([
                "- If **R** improves collapse but adds jitter, it supports the need for governance.",
                "- If **G** reduces collapse with bounded SafetyTax, it supports: *RL-tuned parameters + deterministic gate is a safe control pattern*.",
                "",
                f"Artifacts:\n- `runs.csv`: {runs_csv.name}\n- `summary.json`: {summary_path.name}\n- `config.json`: {cfg_path.name}",
            ])
        )

        report_path = rpt.write(store.path("report.md"))

        return ExperimentResult(
            run_dir=str(run_dir),
            summary={"elapsed_sec": elapsed, "by_env": summary},
            artifacts={"config": str(cfg_path), "runs_csv": str(runs_csv), "summary_json": str(summary_path)},
            report_path=str(report_path),
        )
