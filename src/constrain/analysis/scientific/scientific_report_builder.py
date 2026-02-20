from typing import Dict, Any
import pandas as pd

from .drift_analyzer import DriftAnalyzer
from .intervention_analyzer import InterventionAnalyzer
from .predictive_analyzer import PredictiveAnalyzer
from .phase_transition_analyzer import PhaseTransitionAnalyzer


class ScientificReportBuilder:

    def __init__(self):
        self.analyzers = [
            DriftAnalyzer(),
            InterventionAnalyzer(),
            PredictiveAnalyzer(),
            PhaseTransitionAnalyzer(),
        ]

    def build(self, df: pd.DataFrame) -> Dict[str, Any]:

        report = {}

        for analyzer in self.analyzers:
            result = analyzer.analyze(df)
            report.update(result)

        return report