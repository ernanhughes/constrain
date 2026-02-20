from constrain.data.memory import Memory
from constrain.config import get_config
from constrain.analysis.scientific.scientific_report_builder import ScientificReportBuilder
from constrain.analysis.aggregation.metrics_aggregator import MetricsAggregator

builder = ScientificReportBuilder()
memory = Memory(get_config().db_url)
run_id = "run_18cdc06e"
df = MetricsAggregator.build_run_dataframe(memory, run_id)
report = builder.build(df)
df = df.sort_values(["problem_id", "iteration"])
print(df)
