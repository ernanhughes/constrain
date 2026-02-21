import numpy as np
import torch


def flatten_numeric_dict(
    d: dict,
    parent_key: str = "",
    sep: str = ".",
) -> dict:

    flat = {}

    def coerce_float(v):
        if isinstance(v, (int, float)):
            return float(v)

        if isinstance(v, np.generic):
            return float(v.item())

        if isinstance(v, np.ndarray):
            if v.size == 1:
                return float(v.reshape(-1)[0])
            return float(np.mean(v))

        if isinstance(v, torch.Tensor):
            if v.numel() == 1:
                return float(v.item())
            return float(v.mean().item())

        try:
            return float(v)
        except Exception:
            return None

    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k

        if isinstance(v, dict):
            flat.update(flatten_numeric_dict(v, new_key, sep=sep))

        elif isinstance(v, (list, tuple)):
            # numeric list → index
            for i, item in enumerate(v):
                val = coerce_float(item)
                if val is not None:
                    flat[f"{new_key}[{i}]"] = val

        else:
            val = coerce_float(v)
            if val is not None:
                flat[new_key] = val

    return flat
