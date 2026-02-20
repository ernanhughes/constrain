# constrain/data/orm/derived_metrics.py

from __future__ import annotations

from sqlalchemy import Boolean, Column, Float, Index, Integer, Text

from constrain.data.base import Base


class DerivedMetricsORM(Base):
    """
    Aggregated per-problem dynamic metrics.
    One row per (run_id, problem_id).
    """

    __tablename__ = "derived_metrics"

    id = Column(Integer, primary_key=True, autoincrement=True)

    # Identity
    run_id = Column(Text, nullable=False, index=True)
    problem_id = Column(Integer, nullable=False, index=True)

    # ---------------------------
    # Energy Dynamics
    # ---------------------------

    mean_energy = Column(Float, nullable=True)
    max_energy = Column(Float, nullable=True)
    energy_std = Column(Float, nullable=True)

    drift_slope = Column(Float, nullable=True)
    energy_auc = Column(Float, nullable=True)  # Optional trapezoid AUC

    first_drift_iteration = Column(Integer, nullable=True)
    first_unstable_iteration = Column(Integer, nullable=True)
    collapse_iteration = Column(Integer, nullable=True)

    collapse_detected = Column(Boolean, nullable=False, default=False)

    # ---------------------------
    # Recovery Metrics
    # ---------------------------

    intervention_count = Column(Integer, nullable=False, default=0)

    recovery_delta = Column(Float, nullable=True)
    recovered = Column(Boolean, nullable=True)

    # ---------------------------
    # Accuracy Outcomes
    # ---------------------------

    initial_accuracy = Column(Float, nullable=True)
    final_accuracy = Column(Float, nullable=True)
    accuracy_recovered = Column(Boolean, nullable=True)

    # ---------------------------
    # Signal Features
    # ---------------------------

    max_foreign_ratio = Column(Float, nullable=True)
    mean_repetition = Column(Float, nullable=True)

    __table_args__ = (
        Index("ix_derived_run_problem", "run_id", "problem_id"),
    )
