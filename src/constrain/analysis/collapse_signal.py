from constrain.analysis.aggregation.metrics_aggregator import MetricsAggregator
from constrain.analysis.stage3.engine.signal_modeling_engine import \
    SignalModelingEngine


class CollapseSignalDiscovery:

    def __init__(self, memory):
        self.memory = memory
        self.engine = SignalModelingEngine()

    def analyze(self, run_id: str):

        df = MetricsAggregator.build_run_dataframe(self.memory, run_id)

        df["collapse_label"] = (df["phase"] == "collapse").astype(int)

        return self.engine.run(
            df=df,
            target_col="collapse_label",
            exclude_cols=[
                "step_id",
                "run_id",
                "problem_id",
                "policy_action",
                "phase",
                "accuracy",
                "correctness",
                "extracted_answer",
            ],
        )