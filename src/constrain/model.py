# model.py

import requests

from .config import get_config


def call_model(prompt, temperature):

    cfg = get_config()

    response = requests.post(
        cfg.ollama_url,
        json={
            "model": cfg.model_name,
            "prompt": prompt,
            "temperature": temperature,
            "stream": False
        },
        timeout=120
    )

    response.raise_for_status()
    return response.json()["response"]
