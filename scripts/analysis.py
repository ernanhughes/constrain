import json

from constrain.config import get_config
from constrain.data.memory import Memory

from constrain.analysis.aggregation.metrics_aggregator import MetricsAggregator
from constrain.analysis.stage2.application_evaluator import ApplicationEvaluator
from constrain.analysis.scientific.scientific_report_builder import ScientificReportBuilder
from constrain.analysis.stage3.signal_discovery_service import SignalDiscoveryService


def run_analysis(run_id: str):

    config = get_config()
    memory = Memory(config.db_url)

    print("\n🔍 Building run dataframe...")
    df = MetricsAggregator.build_run_dataframe(memory, run_id)

    print(f"Loaded {len(df)} rows.")

    # ---------------------------------------------------------
    # Stage 2 — Application Evaluation
    # ---------------------------------------------------------

    print("\n📊 Stage 2 — Application Evaluation")
    evaluator = ApplicationEvaluator(memory)
    stage2_summary, _ = evaluator.evaluate_run(run_id)

    print(json.dumps(stage2_summary, indent=2))

    # ---------------------------------------------------------
    # Scientific Analysis
    # ---------------------------------------------------------

    print("\n🧪 Scientific Report")
    builder = ScientificReportBuilder()
    scientific_report = builder.build(df)

    print(json.dumps(scientific_report, indent=2))

    # ---------------------------------------------------------
    # Stage 3 — Signal Discovery (Optional)
    # ---------------------------------------------------------

    print("\n🔬 Stage 3 — Signal Discovery")
    try:
        signal_service = SignalDiscoveryService(memory)
        signal_results = signal_service.analyze_run(run_id)

        print(json.dumps(signal_results["model_results"], indent=2))

    except Exception as e:
        print(f"Signal discovery skipped: {e}")


if __name__ == "__main__":
    run_analysis("run_18cdc06e")