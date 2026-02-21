from constrain.analysis.policy_replay import PolicyReplayService
from constrain.data.memory import Memory


def main():

    memory = Memory()

    run_id = input("Enter run_id: ").strip()

    result = PolicyReplayService.replay_run(
        memory=memory,
        run_id=run_id,
        policy_id=5,
        tau_soft=0.28,
        tau_medium=0.32,
        tau_hard=0.35,
    )

    print("\n=== POLICY REPLAY REPORT ===")
    for k, v in result.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
