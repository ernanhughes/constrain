
import json

from constrain.analysis.final_outcome_evaluator import FinalOutcomeEvaluator
from constrain.analysis.metrics_analyzer import MetricsAnalyzer

run_id = "run_18cdc06e"

res = FinalOutcomeEvaluator().evaluate_run(run_id)
print(json.dumps(res, indent=2))

analyzer = MetricsAnalyzer()

report = analyzer.generate_full_report(run_id)
print(json.dumps(report, indent=2))

