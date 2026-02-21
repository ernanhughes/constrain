import numpy as np

try:
    import torch
except Exception:
    torch = None


def flatten_numeric_dict(d: dict, parent_key: str = "", sep: str = ".") -> dict:
    flat = {}

    def coerce_float(v):
        # ---- Explicit None guard ----
        if v is None:
            return None

        # ---- Native numeric ----
        if isinstance(v, (int, float)):
            if v != v:  # NaN
                return None
            return float(v)

        # ---- NumPy scalar ----
        if isinstance(v, np.generic):
            try:
                return float(v.item())
            except Exception:
                return None

        # ---- NumPy array ----
        if isinstance(v, np.ndarray):
            if v.size == 0:
                return None
            try:
                if v.size == 1:
                    return float(v.reshape(-1)[0])
                return float(np.mean(v))
            except Exception:
                return None

        # ---- Torch tensor ----
        if torch is not None and isinstance(v, torch.Tensor):
            if v.numel() == 0:
                return None
            try:
                if v.numel() == 1:
                    return float(v.item())
                return float(v.mean().item())
            except Exception:
                return None

        # ---- Attempt generic float cast ----
        try:
            fv = float(v)
            if fv != fv:  # NaN
                return None
            return fv
        except Exception:
            return None

    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k

        # ---- Nested dict ----
        if isinstance(v, dict):
            flat.update(flatten_numeric_dict(v, new_key, sep))
            continue

        # ---- List / tuple ----
        if isinstance(v, (list, tuple)):
            for i, item in enumerate(v):
                val = coerce_float(item)
                if val is not None:
                    flat[f"{new_key}[{i}]"] = val
            continue

        # ---- Scalar ----
        val = coerce_float(v)
        if val is not None:
            flat[new_key] = val

    return flat
