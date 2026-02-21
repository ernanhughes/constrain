# data/orm/problem_summary.py

from sqlalchemy import (Boolean, Column, Float, ForeignKey, Index, Integer,
                        String)
from sqlalchemy.orm import relationship

from constrain.data.base import Base


class ProblemSummaryORM(Base):
    __tablename__ = "problem_summaries"

    # ---------------------------------------------------------
    # Primary Key
    # ---------------------------------------------------------

    id = Column(Integer, primary_key=True, autoincrement=True)

    # ---------------------------------------------------------
    # Foreign Keys
    # ---------------------------------------------------------

    run_id = Column(String, ForeignKey("runs.run_id"), nullable=False)

    # ---------------------------------------------------------
    # Problem Identifiers
    # ---------------------------------------------------------

    problem_id = Column(Integer, nullable=False)
    policy_id = Column(Integer, nullable=False)
    # ---------------------------------------------------------
    # Outcome Metrics
    # ---------------------------------------------------------

    final_correct = Column(Boolean, nullable=False)

    any_intervention = Column(Boolean, nullable=False)
    num_interventions = Column(Integer, nullable=False)

    # ---------------------------------------------------------
    # Energy Metrics
    # ---------------------------------------------------------

    avg_energy = Column(Float, nullable=False)
    max_energy = Column(Float, nullable=False)

    # ---------------------------------------------------------
    # Recursion Behavior
    # ---------------------------------------------------------

    num_iterations = Column(Integer, nullable=False)

    # ---------------------------------------------------------
    # Intervention Effectiveness
    # ---------------------------------------------------------

    intervention_helped = Column(Boolean, nullable=False)
    intervention_harmed = Column(Boolean, nullable=False)

    # ---------------------------------------------------------
    # Relationships
    # ---------------------------------------------------------

    run = relationship("RunORM", back_populates="problem_summaries")

    # ---------------------------------------------------------
    # Indexes
    # ---------------------------------------------------------

    __table_args__ = (
        Index("idx_problem_summary_run", "run_id"),
        Index("idx_problem_summary_problem", "problem_id"),
    )