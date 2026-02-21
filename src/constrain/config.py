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
- Automatic logging setup on first use
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------
# LOGGING SETUP (called once when config is first loaded)
# ---------------------------------------------------------

def _ensure_logging():
    root = logging.getLogger()
    if not root.handlers:
        handler = logging.StreamHandler(sys.stderr)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        root.addHandler(handler)
        root.setLevel(logging.INFO)
        # Also set our logger to at least INFO
        logger.setLevel(logging.INFO)

        # --- ADD THESE LINES ---
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("urllib3").setLevel(logging.WARNING)
        logging.getLogger("transformers").setLevel(logging.WARNING)
        logging.getLogger("datasets").setLevel(logging.WARNING)
        # -----------------------

        logger.debug("Default logging configured (no prior handlers)")
# ---------------------------------------------------------
# TOML LOADER
# ---------------------------------------------------------

def _read_toml(path: Path) -> dict:
    if not path.exists():
        logger.debug("TOML file %s not found, using empty config", path)
        return {}

    try:
        import tomllib
        with open(path, "rb") as f:
            data = tomllib.load(f)
        logger.debug("Loaded TOML using tomllib from %s", path)
        return data
    except ImportError:
        try:
            import tomli
            with open(path, "rb") as f:
                data = tomli.load(f)
            logger.debug("Loaded TOML using tomli from %s", path)
            return data
        except Exception as e:
            logger.warning("Failed to parse TOML %s: %s", path, e)
            return {}
    except Exception as e:
        logger.warning("Failed to load TOML %s: %s", path, e)
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

    commit = run(["git", "rev-parse", "HEAD"])
    branch = run(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    dirty = run(["git", "status", "--porcelain"])

    logger.debug("Git metadata: commit=%s, branch=%s, dirty=%s", commit, branch, bool(dirty))
    return {
        "git_commit": commit,
        "git_branch": branch,
        "git_dirty": dirty,
    }

# ---------------------------------------------------------
# CONFIG OBJECT
# ---------------------------------------------------------

@dataclass(frozen=True)
class ConstrainConfig:

    # dirs and paths
    home_dir: str
    base_dir: str
    logs_dir: str
    plots_dir: str
    models_dir: str
    reports_dir: str
    learned_model_path: str
    learned_policy_shadow: bool

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

    fast_mode: bool = True
    policy_mode: str = "recursive"
    provider: str = "ollama"

    # Paper metadata
    notes: Optional[str] = None
    git_commit: Optional[str] = None
    git_branch: Optional[str] = None
    git_dirty: Optional[str] = None

    @classmethod
    def from_env(cls) -> "ConstrainConfig":
        # Ensure logging is set up before we start logging
        _ensure_logging()

        logger.debug("Loading configuration from environment and TOML")
        toml_data = _read_toml(Path("constrain.toml"))

        # Helper to log source
        def get_value(key, toml_section=None, env_var=None, default=None):
            value = None
            source = "default"
            if toml_section and toml_section in toml_data:
                val = toml_data[toml_section].get(key)
                if val is not None:
                    value = val
                    source = "TOML"
            if env_var and value is None:
                env_val = os.getenv(env_var)
                if env_val is not None:
                    value = env_val
                    source = "ENV"
            if value is None:
                value = default
                source = "default"
            logger.debug("Config %s = %s (source: %s)", key, value, source)
            return value

        home_dir = Path(
            os.environ.get(
                "CONSTRAIN_HOME",
                Path.home() / ".constrain"
            )
        )

        home_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        base_dir = home_dir / timestamp
        base_dir.mkdir(parents=True, exist_ok=True)
        logs_dir = base_dir / "logs"
        plots_dir = base_dir / "plots"
        models_dir = home_dir / "models"
        reports_dir = base_dir / "reports"
        learned_model_path = models_dir / "learned_policy.joblib"

        # Ensure directories exist
        for d in [
            base_dir,
            logs_dir,
            plots_dir,
            models_dir,
            reports_dir,
        ]:
            d.mkdir(parents=True, exist_ok=True)

        learned_policy_shadow = bool(get_value(
            "learned_policy_shadow", toml_section="experiment", env_var=None,
            default=False
        ))

        # Core
        db_url = get_value(
            "url", toml_section="database", env_var="DATABASE_URL",
            default="sqlite:///experiment.db"
        )
        provider = get_value(
            "provider", toml_section="model", env_var="MODEL_PROVIDER",
            default="ollama"
        )
        model_name = get_value(
            "name", toml_section="model", env_var="MODEL_NAME",
            default="mistral"
        )
        embedding_model = get_value(
            "embedding_model", toml_section="model", env_var="EMBEDDING_MODEL_NAME",
            default="all-MiniLM-L6-v2"
        )
        embedding_db = get_value(
            "embedding_db", toml_section="model", env_var="EMBEDDING_DB",
            default="embedding.db"
        )
        ollama_url = get_value(
            "ollama_url", toml_section="model", env_var="OLLAMA_URL",
            default="http://localhost:11434/api/generate"
        )

        # Experiment
        num_problems = int(get_value(
            "num_problems", toml_section="experiment", env_var=None,
            default=20
        ))
        num_recursions = int(get_value(
            "num_recursions", toml_section="experiment", env_var=None,
            default=6
        ))
        initial_temperature = float(get_value(
            "initial_temperature", toml_section="experiment", env_var=None,
            default=1.1
        ))
        policy_mode = get_value(
            "policy_mode", toml_section="experiment", env_var=None,
            default="recursive"
        )

        # Thresholds
        tau_soft = float(get_value(
            "soft", toml_section="tau", env_var=None,
            default=0.30
        ))
        tau_medium = float(get_value(
            "medium", toml_section="tau", env_var=None,
            default=0.32
        ))
        tau_hard = float(get_value(
            "hard", toml_section="tau", env_var=None,
            default=0.36
        ))

        # Execution Controls
        run_baseline = bool(get_value(
            "run_baseline", toml_section="execution", env_var=None,
            default=True
        ))
        baseline_policy_id = int(get_value(
            "baseline_policy_id", toml_section="execution", env_var=None,
            default=0
        ))
        run_experiment = bool(get_value(
            "run_experiment", toml_section="execution", env_var=None,
            default=True
        ))
        experiment_policy_id = int(get_value(
            "experiment_policy_id", toml_section="execution", env_var=None,
            default=5
        ))
        run_analysis = bool(get_value(
            "run_analysis", toml_section="execution", env_var=None,
            default=True
        ))
        fast_mode = bool(get_value(
            "fast_mode", toml_section="execution", env_var=None,
            default=True
        ))

        # Paper notes
        notes = toml_data.get("paper", {}).get("notes")

        # Git metadata
        git_meta = {}
        if toml_data.get("paper", {}).get("track_git", True):
            git_meta = _get_git_metadata()

        logger.debug("Configuration loaded successfully")

        return cls(
            home_dir=str(home_dir),
            base_dir=str(base_dir),
            logs_dir=str(logs_dir),
            plots_dir=str(plots_dir),
            models_dir=str(models_dir),
            reports_dir=str(reports_dir),
            learned_model_path=str(learned_model_path),
            learned_policy_shadow=learned_policy_shadow,
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
            "home_dir": self.home_dir,
            "base_dir": self.base_dir,
            "logs_dir": self.logs_dir,
            "plots_dir": self.plots_dir,
            "models_dir": self.models_dir,
            "reports_dir": self.reports_dir,
            "db_url": self.db_url,
            "learned_model_path": self.learned_model_path,  
            "learned_policy_shadow": self.learned_policy_shadow,
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
        logger.debug(f"Constrain config loaded: {_config_instance.to_dict()}")

    return _config_instance