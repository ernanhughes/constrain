# stephanie/utils/json_sanitize.py
from __future__ import annotations

import dataclasses
import decimal
import enum
import json
import math
import pathlib
import uuid
from datetime import date, datetime
from datetime import time as dtime
from typing import Any, Iterable, Mapping
import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------
# Core, single-source-of-truth utilities
# -----------------------------------------------------------------------------


def _to_native_scalar(x: Any) -> tuple[bool, Any]:
    """
    Try to convert x into a JSON-safe scalar.
    Returns (converted, value). If converted=False, caller should recurse/handle.
    """
    # Plain Python numbers/bools/str/None
    if isinstance(x, bool):
        return True, x
    if isinstance(x, int):
        return True, x
    if isinstance(x, float):
        return True, (None if (math.isnan(x) or math.isinf(x)) else x)
    if x is None:
        return True, None
    if isinstance(x, str):
        return True, x

    # Decimals → float (use str(x) if exactness is critical)
    if isinstance(x, decimal.Decimal):
        try:
            return True, float(x)
        except Exception:
            return True, str(x)

    # Datetimes → ISO8601
    if isinstance(x, (datetime, date, dtime)):
        if isinstance(x, datetime) and x.tzinfo is None:
            # make explicit if you prefer UTC; adjust if you want to preserve 'naive'
            x = x.replace(tzinfo=None)
        return True, x.isoformat()

    # Enums / UUID / Paths
    if isinstance(x, enum.Enum):
        return True, _to_native_scalar(x.value)[1]
    if isinstance(x, uuid.UUID):
        return True, str(x)
    if isinstance(x, pathlib.Path):
        return True, str(x)

    # Bytes → hex string (or base64 if you prefer)
    if isinstance(x, (bytes, bytearray, memoryview)):
        try:
            return True, x.hex()
        except Exception:
            return True, str(x)

    if isinstance(x, np.generic):
        return True, x.item()

    if isinstance(x, np.ndarray):
        return True, x.tolist()

    # Dataclasses → dict (let recursion handle)
    if dataclasses.is_dataclass(x):
        return True, dataclasses.asdict(x)

    return False, x


def _sanitize_any(obj: Any, *, max_depth: int = 100) -> Any:
    """
    Recursively convert obj into JSON-serializable Python natives.
    """
    if max_depth <= 0:
        return str(obj)

    if isinstance(obj, pd.DataFrame):
        return _sanitize_any(obj.to_dict(orient="records"), max_depth=max_depth - 1)

    if isinstance(obj, pd.Series):
        return _sanitize_any(obj.to_list(), max_depth=max_depth - 1)

    converted, val = _to_native_scalar(obj)
    if converted:
        return val

    # Mappings
    if isinstance(obj, Mapping):
        out = {}
        for k, v in obj.items():
            out[str(k)] = _sanitize_any(v, max_depth=max_depth - 1)
        return out

    # Iterables (list/tuple/set/frozenset/generators)
    if isinstance(obj, Iterable) and not isinstance(obj, (str, bytes, bytearray)):
        return [_sanitize_any(x, max_depth=max_depth - 1) for x in obj]

    # Fallback: best-effort string (last resort)
    try:
        json.dumps(obj)
        return obj
    except Exception:
        return str(obj)


def json_default(obj: Any):
    """
    Default hook for json.dumps that mirrors _to_native_scalar behavior and
    gracefully stringifies unknown objects.
    """
    converted, val = _to_native_scalar(obj)
    if converted:
        return val
    # Dataclasses to dict
    if dataclasses.is_dataclass(obj):
        return dataclasses.asdict(obj)
    # Fallback
    # Ensure all keys are JSON-serializable
    if isinstance(obj, dict):
        return {str(k): json_default(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [json_default(x) for x in obj]
    else:
        return str(obj)


# -----------------------------------------------------------------------------
# Public API (backward-compatible)
# -----------------------------------------------------------------------------


def dumps_safe(obj: Any, **kwargs) -> str:
    """
    json.dumps with full recursive sanitization for NumPy, Decimal, Enum, UUID,
    datetime, and dataclasses. Handles numpy.int64 keys correctly.
    """
    sanitized = sanitize(obj)
    # Avoid duplicate kwargs by only setting defaults if absent
    kwargs = dict(kwargs)  # copy so we don't mutate caller's dict
    kwargs.setdefault("ensure_ascii", False)
    kwargs.setdefault("default", json_default)
    return json.dumps(sanitized, **kwargs)


def dumps_pretty(obj, **kwargs) -> str:
    """Pretty JSON via dumps_safe."""
    kwargs = dict(kwargs)
    kwargs.setdefault("indent", 2)
    return dumps_safe(obj, **kwargs)


def dumps_compact(obj, **kwargs) -> str:
    """Compact JSON via dumps_safe."""
    kwargs = dict(kwargs)
    kwargs.setdefault("separators", (",", ":"))
    return dumps_safe(obj, **kwargs)


def sanitize(obj: Any) -> Any:
    """Return a JSON-serializable Python object (use for JSON/JSONB columns)."""
    return _sanitize_any(obj)


# --- Back-compat shims (delegate to the core implementations) ---


def json_sanitize(obj: Any) -> Any:
    """Deprecated alias for sanitize()."""
    return _sanitize_any(obj)


def sanitize_for_json(obj: Any, *, max_depth: int = 100) -> Any:
    """Deprecated alias for sanitize() with explicit max_depth."""
    return _sanitize_any(obj, max_depth=max_depth)


def safe_json(obj: Any) -> str:
    """
    Deprecated alias for dumps_safe(). Kept for compatibility where a TEXT
    payload is required.
    """
    return dumps_safe(obj)


def to_json_safe(obj: Any) -> Any:
    """
    Deprecated alias that historically converted NumPy types; now just sanitize().
    Useful when an ORM expects a dict/list for JSONB.
    """
    return _sanitize_any(obj)
