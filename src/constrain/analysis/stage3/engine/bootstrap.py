from __future__ import annotations
import numpy as np
from sklearn.metrics import roc_auc_score



class BootstrapCI:

    @staticmethod
    def auc_ci(
        y_true,
        y_probs,
        n_bootstrap: int = 1000,
        seed: int = 42,
    ):
        rng = np.random.RandomState(seed)
        aucs = []

        y_true = np.asarray(y_true)
        y_probs = np.asarray(y_probs)

        for _ in range(n_bootstrap):
            idx = rng.choice(len(y_true), len(y_true), replace=True)

            if len(np.unique(y_true[idx])) < 2:
                continue

            auc = roc_auc_score(y_true[idx], y_probs[idx])
            aucs.append(auc)

        lower = float(np.percentile(aucs, 2.5))
        upper = float(np.percentile(aucs, 97.5))

        return {
            "mean_bootstrap_auc": float(np.mean(aucs)),
            "ci_lower": lower,
            "ci_upper": upper,
        }
    
    @staticmethod
    def compute(metric_fn, y_true, y_pred, n_boot=1000):

        scores = []

        n = len(y_true)

        for _ in range(n_boot):
            idx = np.random.choice(n, n, replace=True)
            score = metric_fn(y_true[idx], y_pred[idx])
            scores.append(score)

        lower = np.percentile(scores, 2.5)
        upper = np.percentile(scores, 97.5)

        return {
            "mean": float(np.mean(scores)),
            "lower_ci": float(lower),
            "upper_ci": float(upper),
        }