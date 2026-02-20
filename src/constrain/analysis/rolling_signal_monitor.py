from constrain.analysis.signal_discovery_service import SignalDiscoveryService


class RollingSignalMonitor:

    def __init__(self, memory, interval_steps: int = 50):
        self.memory = memory
        self.interval_steps = interval_steps
        self.last_checked_count = 0

    def maybe_run(self, run_id: str):
        """
        Run signal discovery if enough new steps accumulated.
        """

        total_steps = self.memory.steps.count(run_id=run_id)

        if total_steps - self.last_checked_count < self.interval_steps:
            return None

        print(f"🔎 Rolling signal discovery triggered at {total_steps} steps")

        service = SignalDiscoveryService(self.memory)
        results = service.analyze_and_persist(run_id)

        self.last_checked_count = total_steps

        return results
