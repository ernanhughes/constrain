from constrain.services.rolling_monitor import RollingSignalMonitor
import pandas as pd

class SignalMonitoringService:

    def __init__(self, monitor=None):
        self.monitor = monitor or RollingSignalMonitor()

    def analyze_run(self, memory, run_id):

        steps = memory.steps.get_by_run(run_id)

        df = pd.DataFrame([s.model_dump() for s in steps])

        return self.monitor.analyze_run(df)