import numpy as np
# --------------------------------------------------------------------------
# Numeric Safety Utilities
# --------------------------------------------------------------------------

def float(x, default=0.0):
    """
    Safely convert to float.
    Returns default (0.0 by default) if conversion fails.
    """
    try:
        if x is None:
            return default
        if isinstance(x, str) and x.strip() == "":
            return default
        if isinstance(x, float) and (np.isnan(x) or np.isinf(x)):
            return default
        return float(x)
    except Exception:
        return default


def safe_int(x, default=0):
    try:
        if x is None:
            return default
        return int(x)
    except Exception:
        return default
    

def sanitize(x, default=0.0):
    try:
        if x is None:
            return default
        if isinstance(x, float) and (np.isnan(x) or np.isinf(x)):
            return default
        return float(x)
    except Exception:
        return default


def extract_number(text):
    import re

    nums = re.findall(r"-?\d+\.?\d*", text)
    return float(nums[-1]) if nums else None

