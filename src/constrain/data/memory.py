from __future__ import annotations

import logging
from typing import Any, Optional, overload

import numpy as np

from constrain.config import get_config
from constrain.data.engine.session import create_session_factory_from_url
from constrain.data.memory_protocol import MemoryProtocol
from constrain.data.stores.calibration_store import CalibrationStore
from constrain.data.stores.derived_metrics_store import DerivedMetricsStore
from constrain.data.stores.embedding_store import EmbeddingStore
from constrain.data.stores.intervention_store import InterventionStore
from constrain.data.stores.metric_store import MetricStore
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
            ("runs", RunStore),
            ("interventions", InterventionStore),
            ("steps", StepStore),
            ("embeddings", EmbeddingStore),
            ("metrics", MetricStore),
            ("derived_metrics", DerivedMetricsStore),  
            ("signals", SignalDiscoveryStore),
            ("calibrations", CalibrationStore),
            ("signal_reports", SignalReportStore),  # Reusing store for reports
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


    @property
    def embedding_service(self):
        if self._embedding_service is None:
            from constrain.services.hf_embedder_service import HFEmbedder
            config = get_config()
            self._embedding_service = HFEmbedder(config.embedding_model, memory=self)
        return self._embedding_service

    def embed(self, texts: list[str]) -> list[np.ndarray]:
        """
        Public embedding API with fast in-memory cache.
        Falls back to DB store if not cached.
        """

        config = get_config()
        provider = config.provider
        model = config.embedding_model

        results: list[np.ndarray] = []
        missing_texts: list[str] = []
        missing_indices: list[int] = []

        # -------------------------------------------------
        # 1️⃣ Check in-memory cache first
        # -------------------------------------------------
        for i, text in enumerate(texts):
            key = (text, model, provider)
            if key in self._embed_cache:
                results.append(self._embed_cache[key])
            else:
                results.append(None)  # placeholder
                missing_texts.append(text)
                missing_indices.append(i)

        # -------------------------------------------------
        # 2️⃣ If any missing, query DB store
        # -------------------------------------------------
        if missing_texts:
            vecs, still_missing_idx = self.embeddings.get(
                missing_texts,
                model=model,
                provider=provider,
            )

            # -------------------------------------------------
            # 3️⃣ Generate embeddings for truly missing
            # -------------------------------------------------
            if still_missing_idx:
                to_generate = [missing_texts[i] for i in still_missing_idx]

                service = self.embedding_service
                generated = service.embed(to_generate)

                self.embeddings.put(
                    texts=to_generate,
                    vecs=np.array(generated),
                    model=model,
                    provider=provider,
                )

                # Re-fetch
                vecs, _ = self.embeddings.get(
                    missing_texts,
                    model=model,
                    provider=provider,
                )

            # -------------------------------------------------
            # 4️⃣ Fill results + update in-memory cache
            # -------------------------------------------------
            for local_i, vec in enumerate(vecs):
                global_i = missing_indices[local_i]
                key = (texts[global_i], model, provider)

                arr = np.array(vec, dtype=np.float32)
                self._embed_cache[key] = arr
                results[global_i] = arr

        return results
