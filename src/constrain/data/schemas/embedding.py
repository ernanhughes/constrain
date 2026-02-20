from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, ConfigDict


class EmbeddingDTO(BaseModel):
    model_config = ConfigDict(extra="forbid", from_attributes=True)

    id: Optional[int] = None

    text: str
    text_hash: str

    model: str
    provider: str

    run_id: Optional[str] = None

    dim: int
    vector: List[float]

    updated_at: float
