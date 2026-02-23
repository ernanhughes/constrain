import warnings

from constrain.analysis.stage2.policy_comparison import compare_policies
from constrain.config import get_config
from constrain.runner import run

warnings.filterwarnings("ignore")


def main():
    seeds = [42, 43, 44]
    policy_ids = [0, 4]  # baseline vs energy-policy

    for seed in seeds:
        for policy_id in policy_ids:
            run(policy_id=policy_id, seed=seed, num_problems=20)

    compare_policies(policy_ids=policy_ids, seeds=seeds)

# def main():
#     config = get_config()

#     if config.run_baseline:
#         run(policy_id=config.baseline_policy_id)

#     if config.run_experiment:
#         run(policy_id=config.experiment_policy_id)

#     compare_policies(
#         policy_ids=[0, 4, 6],
#         seeds=[42, 43, 44]
#     )



if __name__ == "__main__":
    main()
