from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict


@dataclass
class ExperimentResult:
    run_dir: str
    summary: Dict[str, Any]
    artifacts: Dict[str, str]  # artifact_name -> path
    report_path: str


class Experiment:
    name: str = "experiment"

    def run(self, out_root: Path, **kwargs) -> ExperimentResult:
        raise NotImplementedError

    def _make_run_dir(self, out_root: Path, run_name: str) -> Path:
        return out_root / run_name
