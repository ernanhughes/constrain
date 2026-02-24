# constrain/model.py

from typing import Optional

import requests

from constrain.config import get_config


def call_model(
    prompt: str,
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
    **kwargs,
) -> str:
    """
    Generic model wrapper for Ollama.

    Accepts extra kwargs so experiment code never breaks
    if additional parameters are passed (top_p, etc.).
    """

    cfg = get_config()

    payload = {
        "model": cfg.model_name,
        "prompt": prompt,
        "temperature": temperature,
        "stream": False,
    }

    # Ollama uses "num_predict" for max tokens
    if max_tokens is not None:
        payload["num_predict"] = int(max_tokens)

    response = requests.post(
        cfg.ollama_url,
        json=payload,
        timeout=120,
    )

    response.raise_for_status()
    return response.json()["response"]