from constrain.analysis.aggregation.metrics_aggregator import MetricsAggregator
from constrain.analysis.stage3.engine.signal_modeling_engine import \
    SignalModelingEngine


class EscalationSignalDiscovery:

    def __init__(self, memory):
        self.memory = memory
        self.engine = SignalModelingEngine()

    def analyze(self, run_id: str):

        df = MetricsAggregator.build_run_dataframe(self.memory, run_id)
        df = df.sort_values(["problem_id", "iteration"])

        df["phase_next"] = df.groupby("problem_id")["phase_value"].shift(-1)
        df["escalation"] = (df["phase_next"] > df["phase_value"]).astype(int)

        df = df.dropna()

        return self.engine.run(
            df=df,
            target_col="escalation",
            exclude_cols=[
                "step_id",
                "run_id",
                "problem_id",
                "phase",
                "phase_next",
                "correctness",
                "accuracy",
                "extracted_answer",
                "phase_value",
            ],
        )