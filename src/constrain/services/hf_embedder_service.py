import logging
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import logging as hf_logging

from constrain.data.memory import Memory

hf_logging.set_verbosity_error()
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)


class HFEmbedder:
    """
    HuggingFace embedder that delegates storage
    to an EmbeddingBackend.
    """

    name: str = "HFEmbedder"

    def __init__(
        self,
        model_name: str,
        memory: Memory,
    ):
        self.model_name = model_name
        self.memory = memory
        self.model = SentenceTransformer(model_name)

        import warnings
        warnings.filterwarnings(
            "ignore",
            message=".*embeddings.position_ids.*"
        )

        # Simple in-memory cache (hashable key)
        self._memory_cache: dict[tuple[str, ...], np.ndarray] = {}

    # -------------------------------------------------
    # RAW embedding (no backend interaction)
    # -------------------------------------------------
    def _embed_raw(self, texts: List[str]) -> np.ndarray:
        vecs = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=False,
            show_progress_bar=False,
        ).astype(np.float32)

        if vecs.ndim == 1:
            vecs = vecs.reshape(1, -1)
        elif vecs.ndim != 2:
            raise ValueError(f"Unexpected embedding shape: {vecs.shape}")

        return vecs

    # -------------------------------------------------
    # Public embed (backend-aware)
    # -------------------------------------------------
    def embed(self, texts: List[str]) -> np.ndarray:
        vecs = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=False,
            show_progress_bar=False,
        ).astype(np.float32)

        if vecs.ndim == 1:
            vecs = vecs.reshape(1, -1)
        elif vecs.ndim != 2:
            raise ValueError(f"Unexpected embedding shape: {vecs.shape}")

        return vecs

    def dimension(self) -> int:
        return self.model.get_sentence_embedding_dimension()

