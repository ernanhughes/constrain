import pandas as pd
from constrain.runner import run
from constrain.analysis.application_evaluator import ApplicationEvaluator
from constrain.data.memory import Memory


def compare_policies(policy_ids):

    results = []

    for pid in policy_ids:

        print(f"\n🚀 Running Policy {pid}")
        run_id = run(policy_id=pid)

        memory = Memory()
        evaluator = ApplicationEvaluator(memory)
        summary, _ = evaluator.evaluate_run(run_id)

        summary["policy_id"] = pid
        results.append(summary)

    df = pd.DataFrame(results)

    print("\n==============================")
    print("Policy Comparison Results")
    print("==============================")
    print(df)

    return df