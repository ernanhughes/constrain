def compare_policies(policy_ids):

    results = []

    for pid in policy_ids:
        run_id = run(policy_id=pid)
        evaluator = ApplicationEvaluator(memory)
        summary, _ = evaluator.evaluate_run(run_id)
        summary["policy_id"] = pid
        results.append(summary)

    return pd.DataFrame(results)