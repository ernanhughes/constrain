from __future__ import annotations

from sqlalchemy import Column, Float, Integer, Text
from sqlalchemy.orm import relationship

from constrain.data.base import Base
from constrain.data.orm.collapse_signal import CollapseSignalORM


class ExperimentORM(Base):
    __tablename__ = "experiments"

    id = Column(Integer, primary_key=True, autoincrement=True)

    # Experiment metadata
    experiment_name = Column(Text, nullable=False)
    experiment_type = Column(Text, nullable=False)

    # Configuration
    policy_ids = Column(Text, nullable=False)
    seeds = Column(Text, nullable=False)
    num_problems = Column(Integer, nullable=True)
    num_recursions = Column(Integer, nullable=True)
    initial_temperature = Column(Float, nullable=True)

    # Timing
    start_time = Column(Float, nullable=False)
    end_time = Column(Float, nullable=True)
    status = Column(Text, nullable=False)

    # Results summary
    results_summary = Column(Text, nullable=True)

    # Notes / metadata
    notes = Column(Text, nullable=True)
    git_commit = Column(Text, nullable=True)

    # Relationships
    runs = relationship("RunORM", back_populates="experiment")
    collapse_signals = relationship(
        "CollapseSignalORM",
        back_populates="experiment",
        cascade="all, delete-orphan",
        lazy="dynamic",
    )