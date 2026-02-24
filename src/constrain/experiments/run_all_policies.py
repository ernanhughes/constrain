from constrain.core.runner import run

def main():
    seed = 42
    policy_ids = [0, 1, 2, 5, 6, 7, 8, 9, 10]  # baseline vs energy-policy
    num_problems = 20
    num_recursions = 6
    for policy_id in policy_ids:
        run(policy_id=policy_id, seed=seed, num_problems=num_problems, num_recursions=num_recursions)

if __name__ == "__main__":
    main()
