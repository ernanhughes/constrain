import shap
import numpy as np


class ShapExplainer:

    @staticmethod
    def explain(model, X, max_samples: int = 500):

        if len(X) > max_samples:
            X = X.sample(max_samples, random_state=42)

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)

        mean_abs_shap = np.abs(shap_values).mean(axis=0)

        return sorted(
            zip(X.columns, mean_abs_shap),
            key=lambda x: x[1],
            reverse=True,
        )