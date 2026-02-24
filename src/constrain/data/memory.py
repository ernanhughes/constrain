from __future__ import annotations

import logging
from typing import Any, Optional, overload

import numpy as np

from constrain.config import get_config
from constrain.data.engine.session import create_session_factory_from_url
from constrain.data.memory_protocol import MemoryProtocol
from constrain.data.stores.calibration_store import CalibrationStore
from constrain.data.stores.collapse_signal_store import CollapseSignalStore
from constrain.data.stores.derived_metrics_store import DerivedMetricsStore
from constrain.data.stores.embedding_store import EmbeddingStore
from constrain.data.stores.experiment_store import ExperimentStore
from constrain.data.stores.intervention_store import InterventionStore
from constrain.data.stores.metric_store import MetricStore
from constrain.data.stores.policy_evaluation_store import PolicyEvaluationStore
from constrain.data.stores.policy_event_store import PolicyEventStore
from constrain.data.stores.problem_summary_store import ProblemSummaryStore
from constrain.data.stores.reasoning_state_store import ReasoningStateSnapshotStore
from constrain.data.stores.recovery_experiment_store import \
    RecoveryExperimentStore
from constrain.data.stores.run_store import RunStore
from constrain.data.stores.signal_discovery_store import SignalDiscoveryStore
from constrain.data.stores.signal_report_store import SignalReportStore
from constrain.data.stores.step_store import StepStore

logger = logging.getLogger(__name__)

class Memory(MemoryProtocol):
    """Service locator providing typed access to all Verity stores."""

    def __init__(self, db_url: Optional[str] = None, extra_stores: Optional[list] = None):
        """
        Initialize memory with database connection.

        Args:
            db_url: Optional database URL. If not provided, uses config.db_url
            extra_stores: Optional list of additional store classes to register
        """
        # Resolve DB URL from config if not provided
        config = get_config()
        resolved_db_url = db_url or config.db_url

        # Create session factory
        self.session_maker = create_session_factory_from_url(resolved_db_url)
        self.backend = "postgres"
        if resolved_db_url.startswith("sqlite:"):
            self.backend = "sqlite"

        self._stores: dict[str, Any] = {}

        # Register core stores with consistent naming
        self._register_core_stores()

        # Register extra stores if provided
        if extra_stores:
            for store_class in extra_stores:
                # Derive name from class (e.g., "MyCustomStore" -> "my_custom")
                name = self._derive_store_name(store_class)
                self.register_store(name, store_class(self.session_maker, memory=self))

        # Engine placeholders (explicit, not generic)
        self._embedding_service = None
        self._embed_cache: dict[tuple[str, str, str], np.ndarray] = {}

    def _register_core_stores(self):
        """Register all core Verity stores with standardized names."""
        core_stores = [
            ("calibrations", CalibrationStore),
            ("collapse_signals", CollapseSignalStore),
            ("derived_metrics", DerivedMetricsStore),  
            ("embeddings", EmbeddingStore),
            ("experiments", ExperimentStore),
            ("interventions", InterventionStore),
            ("metrics", MetricStore),
            ("policy_events", PolicyEventStore),
            ("policy_evaluations", PolicyEvaluationStore),
            ("problem_summaries", ProblemSummaryStore),
            ("reasoning_state_snapshots", ReasoningStateSnapshotStore),
            ("recovery_experiments", RecoveryExperimentStore),
            ("runs", RunStore),
            ("signals", SignalDiscoveryStore),
            ("signal_reports", SignalReportStore), 
            ("steps", StepStore),
        ]
        for name, store_class in core_stores:
            self.register_store(name, store_class(self.session_maker, memory=self))


    def _derive_store_name(self, store_class) -> str:
        """Derive store name from class name (e.g., 'GoalStreamStore' -> 'goal_stream')."""
        class_name = store_class.__name__
        if class_name.endswith("Store"):
            class_name = class_name[:-5]
        # Convert CamelCase to snake_case
        import re

        name = re.sub(r"(?<!^)(?=[A-Z])", "_", class_name).lower()
        return name

    def register_store(self, name: str, store):
        """Register a store with explicit name."""
        if name in self._stores:
            logger.error("Store registration failed: %s already exists", name)
            raise ValueError(f"A store named '{name}' is already registered.")
        self._stores[name] = store
        logger.debug("StoreRegistered: %s", name)

    @overload
    def __getattr__(self, name: str) -> Any: ...

    def __getattr__(self, name: str):
        """Provide dot-access to stores with full type checking."""
        if name in self._stores:
            return self._stores[name]
        raise AttributeError(f"'Memory' has no attribute '{name}'")

    def get(self, name: str) -> Optional[Any]:
        """Get store by name (alternative to dot-access)."""
        return self._stores.get(name)
