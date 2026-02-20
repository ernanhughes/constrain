import joblib
import os
import time


class ModelRegistry:

    def __init__(self, base_path="model_registry"):
        self.base_path = base_path
        os.makedirs(base_path, exist_ok=True)

    def save(self, model, metadata: dict):

        timestamp = int(time.time())
        filename = f"model_{timestamp}.joblib"
        path = os.path.join(self.base_path, filename)

        joblib.dump({
            "model": model,
            "metadata": metadata,
        }, path)

        return path