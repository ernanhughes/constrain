from __future__ import annotations

from typing import Any, Dict

import pandas as pd


class ScientificAnalyzer:
    """
    Base class for scientific validation modules.
    """

    def analyze(self, df: pd.DataFrame) -> Dict[str, Any]:
        raise NotImplementedError