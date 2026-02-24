from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

from constrain.reporting.artifact_store import ArtifactStore
from constrain.reporting.markdown_report import MarkdownReport
from constrain.experiments.base import Experiment, ExperimentResult


@dataclass
class AblationConfig:
    steps: int = 2000
    batch_size: int = 32
    schedule_type: str = "blocks"
    block_size: int = 32
    mixture_alpha: float = 0.2
    seeds: Tuple[int, ...] = (0, 1, 2)


class Paper2AblationExperiment(Experiment):
    name = "paper2_ablation"

    def run(self, out_root: Path, cfg: AblationConfig = AblationConfig()) -> ExperimentResult:
        run_dir = self._make_run_dir(out_root, f"{self.name}_{int(time.time())}")
        store = ArtifactStore(run_dir)

        # NOTE: If your repo already has constrain.experiments.experiment_runner, this will use it.
        # Otherwise, replace this import with your runner module.
        from constrain.experiments.experiment_runner import (
            ConstrainExperimentRunner, ExperimentConfig)

        variants = [
            ("M0_baseline", ExperimentConfig(
                name="M0_baseline", optimizer_type="adamw", zmg_enabled=False, leakage=1.0,
                num_steps=cfg.steps, batch_size=cfg.batch_size,
                schedule_type=cfg.schedule_type, block_size=cfg.block_size, mixture_alpha=cfg.mixture_alpha
            )),
            ("M2_langevin", ExperimentConfig(
                name="M2_langevin", optimizer_type="langevin", zmg_enabled=False, leakage=1.0,
                num_steps=cfg.steps, batch_size=cfg.batch_size,
                schedule_type=cfg.schedule_type, block_size=cfg.block_size, mixture_alpha=cfg.mixture_alpha
            )),
            ("M4_langevin_zmg", ExperimentConfig(
                name="M4_langevin_zmg", optimizer_type="langevin", zmg_enabled=True, leakage=1.0,
                num_steps=cfg.steps, batch_size=cfg.batch_size,
                schedule_type=cfg.schedule_type, block_size=cfg.block_size, mixture_alpha=cfg.mixture_alpha
            )),
        ]

        all_rows: List[Dict[str, Any]] = []
        summaries: Dict[str, Any] = {}

        t0 = time.time()
        for label, exp_cfg in variants:
            for seed in cfg.seeds:
                exp_cfg.seed = seed
                runner = ConstrainExperimentRunner(exp_cfg)
                res = runner.run()
                row = {
                    "variant": label,
                    "seed": seed,
                    "collapse_events": res.collapse_events,
                    "stability_per_step": res.stability_per_step,
                    "uncertainty_drop_per_step": res.uncertainty_drop_per_step,
                    "zmg_ratio": res.final_zmg_metrics.get("drift_reduction_ratio", 0.0),
                    "total_time": res.total_time,
                }
                all_rows.append(row)

        elapsed = time.time() - t0

        def mean(xs): return sum(xs) / max(1, len(xs))

        for label, _ in variants:
            rows = [r for r in all_rows if r["variant"] == label]
            summaries[label] = {
                "collapse_events_mean": mean([r["collapse_events"] for r in rows]),
                "stability_mean": mean([r["stability_per_step"] for r in rows]),
                "unc_drop_mean": mean([r["uncertainty_drop_per_step"] for r in rows]),
                "zmg_ratio_mean": mean([r["zmg_ratio"] for r in rows]),
            }

        cfg_path = store.write_json("config.json", asdict(cfg))
        runs_csv = store.write_csv("runs.csv", all_rows)
        summary_path = store.write_json("summary.json", {"elapsed_sec": elapsed, "variants": summaries})

        rpt = MarkdownReport("Paper 2 – Core Ablation (M0 vs M2 vs M4)")
        rpt.add_meta(experiment=self.name, elapsed_sec=elapsed, run_dir=str(run_dir))
        rpt.add_section(
            "What this experiment tests",
            "\n".join([
                "- **M0**: baseline optimizer, no Langevin, no ZMG",
                "- **M2**: adds **Underdamped Langevin** dynamics",
                "- **M4**: adds **ZMG** projection (drift control)",
                "",
                "**Hypothesis:** M2 reduces collapse vs M0; M4 reduces drift vs M2 (zmg_ratio < 1.0) and further reduces collapse under non-IID schedules."
            ])
        )

        table = [
            "| Variant | Collapse↓ | Stability↑ | UncDrop/Step↓ | ZMG Ratio↓ |",
            "|---|---:|---:|---:|---:|",
        ]
        for label in ["M0_baseline", "M2_langevin", "M4_langevin_zmg"]:
            s = summaries[label]
            table.append(
                f"| {label} | {s['collapse_events_mean']:.2f} | {s['stability_mean']:.4f} | {s['unc_drop_mean']:.6f} | {s['zmg_ratio_mean']:.4f} |"
            )
        rpt.add_section("Results (mean over seeds)", "\n".join(table))

        rpt.add_section(
            "Why this proves something",
            "\n".join([
                "- M2 < M0 on collapses supports *stochastic second-order dynamics improve stability*.",
                "- M4 has zmg_ratio < 1.0 supports *drift reduction*.",
                "- If M4 < M2 on collapses, supports synthesis: *drift control improves stability under shift*.",
                "",
                f"Artifacts:\n- `runs.csv`: {runs_csv.name}\n- `summary.json`: {summary_path.name}\n- `config.json`: {cfg_path.name}",
            ])
        )

        report_path = rpt.write(store.path("report.md"))

        return ExperimentResult(
            run_dir=str(run_dir),
            summary={"elapsed_sec": elapsed, "variants": summaries},
            artifacts={"config": str(cfg_path), "runs_csv": str(runs_csv), "summary_json": str(summary_path)},
            report_path=str(report_path),
        )
