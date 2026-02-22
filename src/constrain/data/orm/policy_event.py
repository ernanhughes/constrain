# constrain/data/orm/policy_event.py

from __future__ import annotations

from sqlalchemy import Column, Float, ForeignKey, Integer, Text, Index
from sqlalchemy.orm import relationship

from constrain.data.base import Base


class PolicyEventORM(Base):
    __tablename__ = "policy_events"

    id = Column(Integer, primary_key=True, autoincrement=True)

    run_id = Column(Text, ForeignKey("runs.run_id"), nullable=False)
    step_id = Column(Integer, ForeignKey("steps.id"), nullable=False)

    policy_id = Column(Integer, nullable=False)

    tau_soft = Column(Float, nullable=False)
    tau_medium = Column(Float, nullable=False)
    tau_hard = Column(Float, nullable=False)

    action = Column(Text, nullable=False)
    collapse_probability = Column(Float, nullable=True)

    created_at = Column(Float, nullable=False)

    run = relationship("RunORM")
    step = relationship("StepORM")

    __table_args__ = (
        Index("idx_policy_event_run", "run_id"),
        Index("idx_policy_event_step", "step_id"),
    )