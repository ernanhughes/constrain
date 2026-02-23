from __future__ import annotations
from dataclasses import dataclass


@dataclass
class SafetyTaxResult:
    """
    SafetyTax = (Perf_G - Perf_S) / (Collapse_S - Collapse_G + eps)

    - Negative => gated dominates (better perf AND fewer collapses)
    - Small positive (<1.0) => bounded / acceptable cost
    - Large positive => expensive safety
    - inf => no collapse reduction
    """
    safety_tax: float
    perf_delta: float
    collapse_reduction: float
    dominated: bool
    bounded: bool


def compute_safety_tax(
    perf_g: float,
    perf_s: float,
    collapse_g: float,
    collapse_s: float,
    epsilon: float = 1e-6,
) -> SafetyTaxResult:
    perf_delta = perf_g - perf_s
    collapse_reduction = collapse_s - collapse_g

    if collapse_reduction <= 0:
        return SafetyTaxResult(
            safety_tax=float("inf"),
            perf_delta=perf_delta,
            collapse_reduction=collapse_reduction,
            dominated=False,
            bounded=False,
        )

    safety_tax = perf_delta / (collapse_reduction + epsilon)

    return SafetyTaxResult(
        safety_tax=float(safety_tax),
        perf_delta=float(perf_delta),
        collapse_reduction=float(collapse_reduction),
        dominated=safety_tax < 0,
        bounded=(safety_tax < 1.0) or (safety_tax < 0),
    )


def format_safety_tax(r: SafetyTaxResult) -> str:
    if r.safety_tax == float("inf"):
        return "∞ (no collapse reduction)"
    if r.dominated:
        return f"{r.safety_tax:.4f} ✅ (dominates)"
    if r.bounded:
        return f"{r.safety_tax:.4f} ✓ (bounded)"
    return f"{r.safety_tax:.4f} ⚠ (expensive)"
