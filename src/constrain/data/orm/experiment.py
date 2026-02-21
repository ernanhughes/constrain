# constrain/data/orm/experiment.py

from sqlalchemy import Column, Float, ForeignKey, Integer, Text, Boolean
from sqlalchemy.orm import relationship

from constrain.data.base import Base


class ExperimentORM(Base):
    __tablename__ = "experiments"

    id = Column(Integer, primary_key=True, autoincrement=True)

    # Experiment metadata
    experiment_name = Column(Text, nullable=False)
    experiment_type = Column(Text, nullable=False)  # "policy_comparison" | "threshold_sweep" | "ablation"

    # Configuration
    policy_ids = Column(Text, nullable=False)  # JSON list: "[0, 4, 99]"
    seeds = Column(Text, nullable=False)  # JSON list: "[42, 43, 44]"
    num_problems = Column(Integer, nullable=True)
    num_recursions = Column(Integer, nullable=True)

    # Timing
    start_time = Column(Float, nullable=False)
    end_time = Column(Float, nullable=True)
    status = Column(Text, nullable=False)  # "running" | "completed" | "failed"

    # Results summary (stored as JSON for flexibility)
    results_summary = Column(Text, nullable=True)  # JSON blob of aggregated metrics

    # Notes / metadata
    notes = Column(Text, nullable=True)
    git_commit = Column(Text, nullable=True)

    # Relationships
    runs = relationship("RunORM", back_populates="experiment")