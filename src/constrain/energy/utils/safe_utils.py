import numpy as np


def safe_mean(x):
    x = np.asarray(x)
    return float(np.mean(x)) if len(x) > 0 else None

def safe_std(x, default=None):
    x = np.asarray(x)
    return float(np.std(x)) if len(x) > 1 else default

def safe_var(x, default=None):
    x = np.asarray(x)
    return float(np.var(x)) if len(x) > 1 else default

def safe_corr(x, y):
    x = np.asarray(x)
    y = np.asarray(y)
    if len(x) < 2 or len(y) < 2:
        return None
    if safe_std(x) == 0 or safe_std(y) == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])
