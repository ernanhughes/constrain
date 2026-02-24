from __future__ import annotations

import csv
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, List


class ArtifactStore:
    """Small helper to write JSON/CSV artifacts into a run directory."""

    def __init__(self, run_dir: Path):
        self.run_dir = run_dir
        self.run_dir.mkdir(parents=True, exist_ok=True)

    def write_json(self, name: str, obj: Any) -> Path:
        path = self.run_dir / name
        if is_dataclass(obj):
            obj = asdict(obj)
        path.write_text(json.dumps(obj, indent=2), encoding="utf-8")
        return path

    def write_csv(self, name: str, rows: List[Dict[str, Any]]) -> Path:
        path = self.run_dir / name
        if not rows:
            path.write_text("", encoding="utf-8")
            return path
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            for r in rows:
                w.writerow(r)
        return path

    def path(self, name: str) -> Path:
        return self.run_dir / name
