from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List
import json
import datetime


@dataclass
class ReportSection:
    title: str
    body_md: str


class MarkdownReport:
    """
    Tiny, dependency-free Markdown report builder.

    Intended use:
      rpt = MarkdownReport("Title")
      rpt.add_meta(experiment="...", seed=0)
      rpt.add_section("What", "...")
      rpt.write(run_dir/"report.md")
    """

    def __init__(self, title: str):
        self.title = title
        self.sections: List[ReportSection] = []
        self.metadata: Dict[str, Any] = {}

    def add_meta(self, **kwargs):
        self.metadata.update(kwargs)

    def add_section(self, title: str, body_md: str):
        self.sections.append(ReportSection(title=title, body_md=body_md))

    def to_markdown(self) -> str:
        ts = datetime.datetime.utcnow().isoformat() + "Z"
        meta = {"generated_at": ts, **self.metadata}
        meta_md = "```json\n" + json.dumps(meta, indent=2) + "\n```"

        parts = [f"# {self.title}\n", "## Metadata\n", meta_md, "\n"]
        for s in self.sections:
            parts.append(f"## {s.title}\n\n{s.body_md}\n")
        return "\n".join(parts)

    def write(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_markdown(), encoding="utf-8")
        return path
