from __future__ import annotations
import argparse
from pathlib import Path
import time

from constrain.core.io.markdown_report import MarkdownReport

from constrain.experiments.suite.exp_ablation_paper2 import Paper2AblationExperiment, AblationConfig
from constrain.experiments.suite.exp_zmg_sweep_alpha import ZMGSweepExperiment, ZMGSweepConfig
from constrain.experiments.suite.exp_paper3_eval_srg import Paper3SRGEvalExperiment, Paper3EvalConfig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", type=str, default="experiments/runs")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--seeds", type=str, default="0,1,2")
    ap.add_argument("--skip_paper3", action="store_true")
    args = ap.parse_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    seeds = tuple(int(x.strip()) for x in args.seeds.split(",") if x.strip())

    reports = []
    t0 = time.time()

    # Paper 2 ablation
    r2 = Paper2AblationExperiment().run(out_root, cfg=AblationConfig(steps=args.steps, seeds=seeds))
    reports.append(("Paper2 Ablation", r2.report_path))

    # Alpha sweep
    rA = ZMGSweepExperiment().run(out_root, cfg=ZMGSweepConfig(steps=max(500, args.steps//2), seeds=seeds))
    reports.append(("Paper2 α-sweep", rA.report_path))

    # Paper 3 S/R/G eval (uses lightweight adapter by default)
    if not args.skip_paper3:
        r3 = Paper3SRGEvalExperiment().run(out_root, cfg=Paper3EvalConfig(steps=max(500, args.steps//2), seeds=seeds))
        reports.append(("Paper3 S/R/G eval", r3.report_path))

    # Index
    index = MarkdownReport("Constrain Experiment Suite — Index")
    index.add_meta(out_root=str(out_root), elapsed_sec=(time.time() - t0))

    lines = ["| Experiment | Report |", "|---|---|"]
    for name, path in reports:
        lines.append(f"| {name} | {path} |")
    index.add_section("Reports", "\n".join(lines))

    index_path = out_root / "INDEX.md"
    index.write(index_path)

    print(f"Wrote index: {index_path}")
    for name, p in reports:
        print(f"- {name}: {p}")


if __name__ == "__main__":
    main()
