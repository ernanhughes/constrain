# energy.py
from __future__ import annotations

import re
import unicodedata

import numpy as np
from sentence_transformers import util

from constrain.data.memory import Memory

# ======================================================
# Core Energy Computation
# ======================================================

def compute_energy(
    memory: Memory,
    prompt: str,
    current: str,
    previous: str | None = None,
):
    """
    Computes hallucination energy.

    Core definition (paper-aligned):

        grounding_energy  = 1 - cos(current, prompt)
        stability_energy  = 1 - cos(current, previous)
        hallucination_energy = grounding + stability

    Returns a dictionary of core energy metrics only.
    """

    # ---------------------------------------------
    # Embeddings (via Memory abstraction)
    # ---------------------------------------------

    p_vec = np.array(memory.embed([prompt]), dtype=np.float32)
    c_vec = np.array(memory.embed([current]), dtype=np.float32)

    # ---------------------------------------------
    # Grounding Energy
    # ---------------------------------------------

    grounding = 1.0 - float(util.cos_sim(c_vec, p_vec).item())

    # ---------------------------------------------
    # Stability Energy
    # ---------------------------------------------

    stability = 0.0

    if previous:
        prev_vec = np.array(memory.embed([previous]), dtype=np.float32)
        stability = 1.0 - float(util.cos_sim(c_vec, prev_vec).item())

    # ---------------------------------------------
    # Total Energy (Hallucination Energy)
    # ---------------------------------------------

    total_energy = grounding + stability

    return {
        "total_energy": float(total_energy),
        "grounding_energy": float(grounding),
        "stability_energy": float(stability),
    }


# ======================================================
# SIGNAL METRICS
# ======================================================

def compute_foreign_char_ratio(text: str) -> float:
    if not text:
        return 0.0

    foreign_chars = sum(
        1 for c in text
        if ord(c) > 127 and unicodedata.category(c).startswith("L")
    )

    return foreign_chars / len(text)


def compute_ascii_ratio(text: str) -> float:
    if not text:
        return 1.0

    ascii_chars = sum(1 for c in text if ord(c) <= 127)
    return ascii_chars / len(text)


def compute_repetition_score(text: str) -> float:
    words = re.findall(r"\b\w+\b", text.lower())
    if len(words) < 3:
        return 0.0

    trigrams = [' '.join(words[i:i+3]) for i in range(len(words) - 2)]
    if not trigrams:
        return 0.0

    return 1.0 - (len(set(trigrams)) / len(trigrams))
