
from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from constrain.data.stores.signal_report_store import SignalReportStore

if TYPE_CHECKING:
    from constrain.data.stores.calibration_store import CalibrationStore
    from constrain.data.stores.derived_metrics_store import DerivedMetricsStore
    from constrain.data.stores.embedding_store import EmbeddingStore
    from constrain.data.stores.intervention_store import InterventionStore
    from constrain.data.stores.metric_store import MetricStore
    from constrain.data.stores.run_store import RunStore
    from constrain.data.stores.signal_discovery_store import \
        SignalDiscoveryStore
    from constrain.data.stores.step_store import StepStore

class MemoryProtocol(Protocol):
    """Protocol defining all available stores for type checking and IDE support."""
    # Core stores
    runs: RunStore
    steps: StepStore
    interventions: InterventionStore
    embedding: EmbeddingStore
    metrics: MetricStore
    derived_metrics: DerivedMetricsStore
    signals: SignalDiscoveryStore
    calibrations: CalibrationStore
    signal_reports: SignalReportStore 

