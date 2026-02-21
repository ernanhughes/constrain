from constrain.runner import run
from constrain.analysis.stage2.application_evaluator import ApplicationEvaluator
from constrain.config import get_config
from constrain.data.memory import Memory

class PolicyBenchmark:

    def compare(self, policy_ids, seed=42):

        results = []

        for pid in policy_ids:
            run_id = run(policy_id=pid, seed=seed)

            evaluator = ApplicationEvaluator(Memory(get_config().db_url))
            summary, _ = evaluator.evaluate_run(run_id)

            summary["policy_id"] = pid
            results.append(summary)

        return results
