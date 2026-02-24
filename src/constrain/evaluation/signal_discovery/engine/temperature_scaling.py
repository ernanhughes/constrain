from __future__ import annotations

import numpy as np
from scipy.optimize import minimize


class TemperatureScaler:

    def __init__(self):
        self.temperature = 1.0

    def _nll(self, temp, logits, y_true):
        scaled = logits / temp
        probs = 1 / (1 + np.exp(-scaled))
        eps = 1e-12
        loss = -np.mean(
            y_true * np.log(probs + eps)
            + (1 - y_true) * np.log(1 - probs + eps)
        )
        return loss

    def fit(self, logits, y_true):

        result = minimize(
            self._nll,
            x0=[1.0],
            args=(logits, y_true),
            bounds=[(0.05, 10.0)],
        )

        self.temperature = float(result.x[0])
        return self

    def transform(self, logits):
        scaled = logits / self.temperature
        return 1 / (1 + np.exp(-scaled))