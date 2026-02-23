import pandas as pd
from .regime_collapse_detector import RegimeCollapseDetector


class SurvivalAnalyzer:

    def __init__(self):
        self.detector = RegimeCollapseDetector()

    def build_survival_table(
        self,
        df: pd.DataFrame,
        tau_stable: float,
        tau_unstable: float,
        tau_soft: float,
        tau_medium: float,
    ):

        rows = []

        for (run_id, pid), group in df.groupby(["run_id", "problem_id"]):

            group = group.sort_values("iteration")

            probs = group["collapse_probability"].fillna(0).tolist()
            energies = group["total_energy"].tolist()

            t_event = self.detector.detect(
                collapse_probs=probs,
                energies=energies,
                tau_stable=tau_stable,
                tau_unstable=tau_unstable,
                tau_soft=tau_soft,
                tau_medium=tau_medium,
            )

            if t_event is None:
                time = len(group)
                event = 0
            else:
                time = t_event
                event = 1

            rows.append({
                "run_id": run_id,
                "problem_id": pid,
                "time_to_event": time,
                "event": event,
            })

        return pd.DataFrame(rows)