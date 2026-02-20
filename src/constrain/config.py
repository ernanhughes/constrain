# constrain/config.py
"""
Configuration management for Constrain.

Features:
- Loads from constrain.toml
- ENV fallbacks
- Git metadata capture (for reproducibility)
- SQLite-safe
- Immutable config object
- Singleton access
"""

from __future__ import annotations

import logging
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------
# TOML LOADER
# ---------------------------------------------------------

def _read_toml(path: Path) -> dict:
    if not path.exists():
        return {}

    try:
        import tomllib
        return tomllib.loads(path.read_text(encoding="utf-8"))
    except ImportError:
        try:
            import tomli
            with open(path, "rb") as f:
                return tomli.load(f)
        except Exception:
            return {}
    except Exception:
        return {}

# ---------------------------------------------------------
# GIT METADATA
# ---------------------------------------------------------

def _get_git_metadata() -> Dict[str, Optional[str]]:
    def run(cmd):
        try:
            return subprocess.check_output(cmd, stderr=subprocess.DEVNULL).decode().strip()
        except Exception:
            return None

    return {
        "git_commit": run(["git", "rev-parse", "HEAD"]),
        "git_branch": run(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        "git_dirty": run(["git", "status", "--porcelain"]),
    }

# ---------------------------------------------------------
# CONFIG OBJECT
# ---------------------------------------------------------

@dataclass(frozen=True)
class ConstrainConfig:

    # Core
    db_url: str
    model_name: str
    embedding_model: str
    embedding_db: str
    ollama_url: str


    # Experiment
    num_problems: int
    num_recursions: int
    initial_temperature: float


    # Thresholds
    tau_soft: float
    tau_medium: float
    tau_hard: float

    # Execution Controls
    run_baseline: bool
    baseline_policy_id: int
    run_experiment: bool
    experiment_policy_id: int
    run_analysis: bool

    fast_mode: bool = True  # If True  (e.g., reuse prompts you ahve already seen in the memory, skip energy computation, etc.)

    policy_mode: str = "recursive"  # static | recursive | adaptive | dynamic
    provider: str = "ollama"

    # Paper metadata
    notes: Optional[str] = None
    git_commit: Optional[str] = None
    git_branch: Optional[str] = None
    git_dirty: Optional[str] = None


    @classmethod
    def from_env(cls) -> "ConstrainConfig":

        toml_data = _read_toml(Path("constrain.toml"))

        db_url = (
            toml_data.get("database", {}).get("url")
            or os.getenv("DATABASE_URL")
            or "sqlite:///experiment.db"
        )

        provider = (
            toml_data.get("model", {}).get("provider")
            or os.getenv("MODEL_PROVIDER")
            or "ollama"
        )

        model_name = (
            toml_data.get("model", {}).get("name")
            or os.getenv("MODEL_NAME")
            or "mistral"
        )

        embedding_model = (
            toml_data.get("model", {}).get("embedding_model")
            or os.getenv("EMBEDDING_MODEL_NAME")
            or "all-MiniLM-L6-v2"
        )

        embedding_db = (
            toml_data.get("model", {}).get("embedding_db")
            or os.getenv("EMBEDDING_DB")
            or "embedding.db"
        )

        ollama_url = (
            toml_data.get("model", {}).get("ollama_url")
            or os.getenv("OLLAMA_URL")
            or "http://localhost:11434/api/generate"
        )

        num_problems = int(
            toml_data.get("experiment", {}).get("num_problems", 20)
        )

        num_recursions = int(
            toml_data.get("experiment", {}).get("num_recursions", 6)
        )

        initial_temperature = float(
            toml_data.get("experiment", {}).get("initial_temperature", 1.1)
        )

        policy_mode = toml_data.get("experiment", {}).get("policy_mode", "recursive")

        tau_soft = float(
            toml_data.get("tau", {}).get("soft", 0.30)
        )

        tau_medium = float(
            toml_data.get("tau", {}).get("medium", 0.32)
        )

        tau_hard = float(
            toml_data.get("tau", {}).get("hard", 0.36)
        )
        # ---------------------------------------------------------
        # Execution Controls
        # ---------------------------------------------------------

        run_baseline = bool(
            toml_data.get("execution", {}).get("run_baseline", True)
        )

        baseline_policy_id = int(
            toml_data.get("execution", {}).get("baseline_policy_id", 0)
        )

        run_experiment = bool(
            toml_data.get("execution", {}).get("run_experiment", True)
        )

        experiment_policy_id = int(
            toml_data.get("execution", {}).get("experiment_policy_id", 5)
        )

        run_analysis = bool(
            toml_data.get("execution", {}).get("run_analysis", True)
        )

        fast_mode = bool(
            toml_data.get("execution", {}).get("fast_mode", True)
        )


        notes = toml_data.get("paper", {}).get("notes")

        git_meta = {}
        if toml_data.get("paper", {}).get("track_git", True):
            git_meta = _get_git_metadata()

        return cls(
            db_url=db_url,
            model_name=model_name,
            embedding_model=embedding_model,
            embedding_db=embedding_db,
            provider=provider,
            ollama_url=ollama_url,
            num_problems=num_problems,
            num_recursions=num_recursions,
            initial_temperature=initial_temperature,
            policy_mode=policy_mode,
            tau_soft=tau_soft,
            tau_medium=tau_medium,
            tau_hard=tau_hard,
            run_baseline=run_baseline,
            baseline_policy_id=baseline_policy_id,
            run_experiment=run_experiment,
            experiment_policy_id=experiment_policy_id,
            run_analysis=run_analysis,
            notes=notes,
            fast_mode=fast_mode,
            git_commit=git_meta.get("git_commit"),
            git_branch=git_meta.get("git_branch"),
            git_dirty=git_meta.get("git_dirty"),
        )

    # ---------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------

    def ensure_sqlite_dir(self):
        if self.db_url.startswith("sqlite:///"):
            db_path = self.db_url.replace("sqlite:///", "")
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "db_url": self.db_url,
            "model_name": self.model_name,
            "provider": self.provider,
            "embedding_model": self.embedding_model,
            "embedding_db": self.embedding_db,
            "num_problems": self.num_problems,
            "num_recursions": self.num_recursions,
            "initial_temperature": self.initial_temperature,
            "policy_mode": self.policy_mode,
            "tau_soft": self.tau_soft,
            "tau_medium": self.tau_medium,
            "tau_hard": self.tau_hard,
            "baseline_policy_id": self.baseline_policy_id,
            "run_experiment": self.run_experiment,
            "experiment_policy_id": self.experiment_policy_id,
            "run_analysis": self.run_analysis,
            "notes": self.notes,
            "fast_mode": self.fast_mode,
            "git_commit": self.git_commit,
            "git_branch": self.git_branch,
            "git_dirty": bool(self.git_dirty),
            "run_baseline": self.run_baseline,
        }

# ---------------------------------------------------------
# SINGLETON ACCESS
# ---------------------------------------------------------

_config_instance: Optional[ConstrainConfig] = None

def get_config() -> ConstrainConfig:
    global _config_instance

    if _config_instance is None:
        _config_instance = ConstrainConfig.from_env()
        _config_instance.ensure_sqlite_dir()
        logger.info(f"Loaded Constrain config: {_config_instance.to_dict()}")

    return _config_instance
