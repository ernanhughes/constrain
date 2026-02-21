import joblib
import os
import time
from constrain.config import get_config

class ModelRegistry:

    def __init__(self, base_path="model_registry"):
        self.base_path = base_path
        os.makedirs(base_path, exist_ok=True)

    def save(self, model, metadata: dict):

        timestamp = int(time.time())
        get_config().models_dir
        filename = f"{get_config().models_dir}/model_{timestamp}.joblib"
        path = os.path.join(self.base_path, filename)

        joblib.dump({
            "model": model,
            "metadata": metadata,
        }, path)

        return path