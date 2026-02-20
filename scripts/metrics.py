from constrain.data.memory import Memory
from constrain.analysis.metrics_aggregator import MetricsAggregator
from constrain.config import get_config

memory = Memory(get_config().db_url)
run_id = "run_18cdc06e"
df = MetricsAggregator.build_run_dataframe(memory, run_id)
df = df.sort_values(["problem_id", "iteration"])
print(df)
