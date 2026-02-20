from __future__ import annotations

from sqlalchemy import (Column, Float, ForeignKey, Index, Integer, LargeBinary,
                        Text, UniqueConstraint)
from sqlalchemy.orm import relationship

from constrain.data.base import Base


class EmbeddingORM(Base):
    __tablename__ = "embeddings"

    id = Column(Integer, primary_key=True, autoincrement=True)

    # Text identity
    text = Column(Text, nullable=False)
    text_hash = Column(Text, nullable=False)

    # Embedding I
    model = Column(Text, nullable=True)
    provider = Column(Text, nullable=True)

    # Optional experiment link
    run_id = Column(Text, ForeignKey("runs.run_id"), nullable=True)

    # Vector data
    dim = Column(Integer, nullable=False)
    vec = Column(LargeBinary, nullable=False)

    updated_at = Column(Float, nullable=False)

    run = relationship("RunORM", backref="embeddings")

    __table_args__ = (
        UniqueConstraint(
            "text_hash",
            name="uq_embedding_identity",
        ),
        Index("idx_embedding_lookup", "text_hash"),
    )
