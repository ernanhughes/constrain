# energy.py
from __future__ import annotations


import numpy as np

from constrain.energy.geometry.claim_evidence import ClaimEvidenceGeometry
from constrain.energy.embedding.hf_embedder import HFEmbedder
from constrain.energy.embedding.sqlite_embedding_backend import SQLiteEmbeddingBackend
from constrain.config import get_config
from constrain.energy.utils.text_utils import split_into_sentences

# ======================================================
# Core Energy Computation
# ======================================================

# instantiate once at module level
_GEOMETRY = ClaimEvidenceGeometry(top_k=6, rank_r=4)
_EMBEDDER = HFEmbedder(
    model_name=get_config().embedding_model,
    backend=SQLiteEmbeddingBackend(str(get_config().embedding_db)),
)


def compute_energy(prompt, current, reasoning_history):
    # --- 1️⃣ Build evidence text list

    evidence_texts = []

    # Prompt sentences
    prompt_sents = split_into_sentences(prompt)
    evidence_texts.extend(prompt_sents)

    # Accepted reasoning history sentences
    for past_reasoning in reasoning_history:
        sents = split_into_sentences(past_reasoning)
        evidence_texts.extend(sents)

    # Guard: must have at least 1
    if not evidence_texts:
        evidence_texts = [prompt]

    # --- 2️⃣ Embed evidence

    evidence_vecs = _EMBEDDER.embed(evidence_texts)
    evidence_vecs = evidence_vecs / (
        np.linalg.norm(evidence_vecs, axis=1, keepdims=True) + 1e-8
    )

    # --- 3️⃣ Embed claim (current reasoning)

    claim_vec = _EMBEDDER.embed([current])[0]
    claim_vec = claim_vec / (np.linalg.norm(claim_vec) + 1e-8)

    # --- 4️⃣ Compute grounding energy

    res_ground = _GEOMETRY.compute(
        claim_vec=claim_vec,
        evidence_vecs=evidence_vecs,
    )

    grounding_energy = float(res_ground.energy)

    # -------------------------------------------------
    # 3️⃣ Stability Energy (previous ↔ current)
    # -------------------------------------------------

    res_stability = None

    stability_energy = 0.0

    if len(reasoning_history) > 0:

        last_stable = reasoning_history[-1]

        last_sents = split_into_sentences(last_stable)

        if last_sents:
            last_vecs = _EMBEDDER.embed(last_sents)
            last_vecs = last_vecs / (
                np.linalg.norm(last_vecs, axis=1, keepdims=True) + 1e-8
            )

            res_stability = _GEOMETRY.compute(
                claim_vec=claim_vec,   # current
                evidence_vecs=last_vecs,
            )

            stability_energy = float(res_stability.energy)

    total_energy = grounding_energy + stability_energy

    # -------------------------------------------------
    # 4️⃣ Extract ALL Geometry Attributes
    # -------------------------------------------------

    g = res_ground.geometry

    features = {
        # Core
        "total_energy": float(total_energy),
        "grounding_energy": grounding_energy,
        "stability_energy": stability_energy,
        # Raw energy internals
        "explained": float(res_ground.explained),
        "identity_error": float(res_ground.identity_error),
        # Spectral structure
        "sigma1_ratio": float(g.sigma1_ratio),
        "sigma2_ratio": float(g.sigma2_ratio),
        "spectral_sum": float(g.spectral_sum),
        "participation_ratio": float(g.participation_ratio),
        # Rank
        "effective_rank": int(g.effective_rank),
        "used_count": int(g.used_count),
        "entropy_rank": float(g.entropy_rank),
        # Alignment
        "alignment_to_sigma1": float(g.alignment_to_sigma1),
        # Similarity
        "sim_top1": float(g.sim_top1),
        "sim_top2": float(g.sim_top2),
        "sim_margin": float(g.sim_margin),
        # Brittleness
        "sensitivity": float(g.sensitivity),
    }

    return features
