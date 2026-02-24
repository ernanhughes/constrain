from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

from constrain.reporting.artifact_store import ArtifactStore
from constrain.reporting.markdown_report import MarkdownReport
from constrain.experiments.base import Experiment, ExperimentResult


@dataclass
class ZMGSweepConfig:
    steps: int = 2000
    batch_size: int = 32
    seeds: Tuple[int, ...] = (0, 1, 2)
    alphas: Tuple[float, ...] = (0.1, 0.2, 0.5, 1.0, 5.0)
    optimizer_type: str = "langevin"
    leakage: float = 0.9
    gamma: float = 0.1
    temperature: float = 1.0


class ZMGSweepExperiment(Experiment):
    name = "paper2_zmg_alpha_sweep"

    def run(self, out_root: Path, cfg: ZMGSweepConfig = ZMGSweepConfig()) -> ExperimentResult:
        run_dir = self._make_run_dir(out_root, f"{self.name}_{int(time.time())}")
        store = ArtifactStore(run_dir)

        from constrain.experiments.experiment_runner import (
            ConstrainExperimentRunner, ExperimentConfig)

        all_rows: List[Dict[str, Any]] = []
        t0 = time.time()

        for alpha in cfg.alphas:
            for seed in cfg.seeds:
                exp_cfg_zmg = ExperimentConfig(
                    name=f"zmg_alpha_{alpha}",
                    optimizer_type=cfg.optimizer_type,
                    zmg_enabled=True,
                    leakage=cfg.leakage,
                    gamma=cfg.gamma,
                    temperature=cfg.temperature,
                    num_steps=cfg.steps,
                    batch_size=cfg.batch_size,
                    schedule_type="mixture",
                    mixture_alpha=alpha,
                    seed=seed,
                )
                exp_cfg_no = ExperimentConfig(
                    name=f"no_zmg_alpha_{alpha}",
                    optimizer_type=cfg.optimizer_type,
                    zmg_enabled=False,
                    leakage=cfg.leakage,
                    gamma=cfg.gamma,
                    temperature=cfg.temperature,
                    num_steps=cfg.steps,
                    batch_size=cfg.batch_size,
                    schedule_type="mixture",
                    mixture_alpha=alpha,
                    seed=seed,
                )

                res_zmg = ConstrainExperimentRunner(exp_cfg_zmg).run()
                res_no = ConstrainExperimentRunner(exp_cfg_no).run()

                row = {
                    "alpha": alpha,
                    "seed": seed,
                    "collapse_rate_zmg": res_zmg.collapse_events / cfg.steps,
                    "collapse_rate_no_zmg": res_no.collapse_events / cfg.steps,
                    "zmg_ratio": res_zmg.final_zmg_metrics.get("drift_reduction_ratio", 0.0),
                    "stability_zmg": res_zmg.stability_per_step,
                }
                all_rows.append(row)

        elapsed = time.time() - t0

        def mean(xs): return sum(xs) / max(1, len(xs))
        def std(xs):
            if len(xs) < 2:
                return 0.0
            m = mean(xs)
            return (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5

        summaries: Dict[float, Dict[str, Any]] = {}
        for alpha in cfg.alphas:
            rows = [r for r in all_rows if r["alpha"] == alpha]
            summaries[alpha] = {
                "zmg_ratio_mean": mean([r["zmg_ratio"] for r in rows]),
                "zmg_ratio_std": std([r["zmg_ratio"] for r in rows]),
                "collapse_reduction": mean([r["collapse_rate_no_zmg"] - r["collapse_rate_zmg"] for r in rows]),
                "stability_mean": mean([r["stability_zmg"] for r in rows]),
            }

        cfg_path = store.write_json("config.json", asdict(cfg))
        runs_csv = store.write_csv("runs.csv", all_rows)
        summary_path = store.write_json("summary.json", {"elapsed_sec": elapsed, "by_alpha": summaries})

        rpt = MarkdownReport("Paper 2 — ZMG Heterogeneity Sweep (α)")
        rpt.add_meta(experiment=self.name, elapsed_sec=elapsed, run_dir=str(run_dir))

        rpt.add_section(
            "What this experiment tests",
            "\n".join([
                "**Hypothesis:** ZMG reduces gradient drift under non-IID stress; benefit approaches 1.0 as α→∞ (IID).",
                "",
                "We compare ZMG-enabled vs ZMG-disabled runs under mixture schedules with varying α.",
                "Metric: `drift_reduction_ratio = drift_proj / drift_raw` (want < 1.0, and trend → 1.0 as α increases).",
            ])
        )

        table = ["| α | ZMG Ratio↓ | Collapse Reduction↑ | Stability↑ |",
                 "|---|---:|---:|---:|"]
        for alpha in cfg.alphas:
            s = summaries[alpha]
            ratio = f"{s['zmg_ratio_mean']:.4f}±{s['zmg_ratio_std']:.4f}"
            if s["zmg_ratio_mean"] < 1.0:
                ratio += " ✅"
            table.append(f"| {alpha} | {ratio} | {s['collapse_reduction']:.4f} | {s['stability_mean']:.4f} |")
        rpt.add_section("Results (mean ± std over seeds)", "\n".join(table))

        rpt.add_section(
            "Why this proves something",
            "\n".join([
                "- Ratio < 1.0 at low α supports *drift reduction under heterogeneity*.",
                "- If ratio trends upward toward 1.0 as α grows, that is the expected graceful degradation in IID limit.",
                "- Positive collapse reduction at low α supports the stability-via-drift-control story.",
                "",
                f"Artifacts:\n- `runs.csv`: {runs_csv.name}\n- `summary.json`: {summary_path.name}\n- `config.json`: {cfg_path.name}",
            ])
        )

        report_path = rpt.write(store.path("report.md"))

        return ExperimentResult(
            run_dir=str(run_dir),
            summary={"elapsed_sec": elapsed, "by_alpha": summaries},
            artifacts={"config": str(cfg_path), "runs_csv": str(runs_csv), "summary_json": str(summary_path)},
            report_path=str(report_path),
        )
