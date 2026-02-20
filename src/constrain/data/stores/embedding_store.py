from __future__ import annotations

import hashlib
import json
import time
from typing import Any, List, Optional, Tuple

import numpy as np
from sqlalchemy.orm import sessionmaker

from constrain.data.orm.embedding import EmbeddingORM
from constrain.data.schemas.embedding import EmbeddingDTO
from constrain.data.stores.base_store import BaseSQLAlchemyStore


class EmbeddingStore(BaseSQLAlchemyStore[EmbeddingDTO]):
    orm_model = EmbeddingORM
    default_order_by = "updated_at"

    def __init__(self, sm: sessionmaker, memory: Optional[Any] = None):
       super().__init__(sm, memory)
       self.name = "embeddings"

    # -------------------------------------------------
    # Identity Helpers
    # -------------------------------------------------

    @staticmethod
    def _hash_text(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    @staticmethod
    def _hash_config(config: dict) -> str:
        return hashlib.sha256(
            json.dumps(config, sort_keys=True).encode("utf-8")
        ).hexdigest()

    @staticmethod
    def _to_dto(row: EmbeddingORM) -> EmbeddingDTO:
        vec = np.frombuffer(row.vec, dtype=np.float32)
        return EmbeddingDTO(
            id=row.id,
            text=row.text,
            text_hash=row.text_hash,
            model=row.model,
            provider=row.provider,
            run_id=row.run_id,
            dim=row.dim,
            vector=vec.tolist(),
            updated_at=row.updated_at,
        )

    # -------------------------------------------------
    # Fetch
    # -------------------------------------------------

    def get(
        self,
        texts: List[str],
        *,
        model: str,
        provider: str,
    ) -> Tuple[List[np.ndarray | None], List[int]]:

        if not texts:
            return [], []

        hashes = [self._hash_text(t) for t in texts]

        def op(s):
            return (
                s.query(EmbeddingORM)
                .filter(
                    EmbeddingORM.model == model,
                    EmbeddingORM.provider == provider,
                    EmbeddingORM.text_hash.in_(hashes),
                )
                .all()
            )

        rows = self._run(op)

        results = {r.text_hash: r for r in rows}

        vecs: List[np.ndarray | None] = []
        missing_idx: List[int] = []

        for i, h in enumerate(hashes):
            row = results.get(h)
            if not row:
                vecs.append(None)
                missing_idx.append(i)
                continue

            v = np.frombuffer(row.vec, dtype=np.float32)
            if v.shape[0] != row.dim:
                vecs.append(None)
                missing_idx.append(i)
                continue

            vecs.append(v)

        return vecs, missing_idx

    # -------------------------------------------------
    # Insert / Upsert
    # -------------------------------------------------

    def put(
        self,
        texts: List[str],
        vecs: np.ndarray,
        *,
        model: str,
        provider: str,
        run_id: Optional[str] = None,
    ) -> List[EmbeddingDTO]:

        now = time.time()

        def op(s):
            out = []

            for text, vec in zip(texts, vecs):
                text_hash = self._hash_text(text)

                existing = (
                    s.query(EmbeddingORM)
                    .filter(
                        EmbeddingORM.text_hash == text_hash,
                        EmbeddingORM.model == model,
                        EmbeddingORM.provider == provider,
                    )
                    .first()
                )

                if existing:
                    existing.vec = vec.astype(np.float32).tobytes()
                    existing.dim = int(vec.shape[0])
                    existing.updated_at = now
                    obj = existing
                else:
                    obj = EmbeddingORM(
                        text=text,
                        text_hash=text_hash,
                        model=model,
                        provider=provider,
                        run_id=run_id,
                        dim=int(vec.shape[0]),
                        vec=vec.astype(np.float32).tobytes(),
                        updated_at=now,
                    )
                    s.add(obj)

                s.flush()
                out.append(self._to_dto(obj))

            return out

        return self._run(op)

    def cosine_search(
        self,
        query_vec: np.ndarray,
        *,
        model: str,
        provider: str,
        top_k: int = 5,
    ):
        def op(s):
            rows = (
                s.query(EmbeddingORM)
                .filter(
                    EmbeddingORM.model == model,
                    EmbeddingORM.provider == provider,
                )
                .all()
            )
            return rows

        rows = self._run(op)

        sims = []
        for r in rows:
            vec = np.frombuffer(r.vec, dtype=np.float32)
            score = float(np.dot(query_vec, vec) / (
                np.linalg.norm(query_vec) * np.linalg.norm(vec) + 1e-8
            ))
            sims.append((score, r))

        sims.sort(key=lambda x: x[0], reverse=True)

        return [(score, self._to_dto(r)) for score, r in sims[:top_k]]
