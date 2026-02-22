# constrain/data/orm/stage2_evaluation.py

from sqlalchemy import Column, Float, ForeignKey, Integer, Text, Boolean
from sqlalchemy.orm import relationship

from constrain.data.base import Base


class PolicyEvaluationORM(Base):
    __tablename__ = "policy_evaluations"

    id = Column(Integer, primary_key=True, autoincrement=True)

    # Link to source run
    run_id = Column(Text, ForeignKey("runs.run_id"), nullable=False)
    problem_id = Column(Integer, nullable=False)

    # Policy identification
    policy_id = Column(Integer, nullable=False)
    seed = Column(Integer, nullable=True)
    experiment_id = Column(Integer, ForeignKey("experiments.id"), nullable=True)

    # Outcome metrics
    final_correct = Column(Boolean, nullable=False)
    any_intervention = Column(Boolean, nullable=False)
    intervention_helped = Column(Boolean, nullable=False)
    intervention_harmed = Column(Boolean, nullable=False)

    # Aggregated metrics
    num_iterations = Column(Integer, nullable=False)
    avg_energy = Column(Float, nullable=False)
    max_energy = Column(Float, nullable=False)

    # Thresholds used (for sweep analysis)
    tau_soft = Column(Float, nullable=False)
    tau_medium = Column(Float, nullable=False)
    tau_hard = Column(Float, nullable=False)

    # Metadata
    created_at = Column(Float, nullable=False)

    # Relationships
    run = relationship("RunORM")
    experiment = relationship("ExperimentORM")