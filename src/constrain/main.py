import warnings

from constrain.analysis.compare_policies import compare_policies

from constrain.config import get_config
from constrain.runner import run

warnings.filterwarnings("ignore")


def main():
    config = get_config()

    if config.run_baseline:
        run(policy_id=config.baseline_policy_id)

    if config.run_experiment:
        run(policy_id=config.experiment_policy_id)

    compare_policies([0, 4])



if __name__ == "__main__":
    main()
