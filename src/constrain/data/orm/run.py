from sqlalchemy import Column, Float, ForeignKey, Integer, Text
from sqlalchemy.orm import relationship

from constrain.data.base import Base


class RunORM(Base):
    __tablename__ = "runs"

    problem_summaries = relationship(
        "ProblemSummaryORM",
        back_populates="run",
        cascade="all, delete-orphan",
    )

    run_id = Column(Text, primary_key=True)

    model_name = Column(Text, nullable=False)
    initial_temperature = Column(Float, nullable=False)

    num_problems = Column(Integer, nullable=False)
    num_recursions = Column(Integer, nullable=False)

    # ------------------------------
    # Energy thresholds
    # ------------------------------

    tau_soft = Column(Float, nullable=False)
    tau_medium = Column(Float, nullable=False)
    tau_hard = Column(Float, nullable=False)

    tau_soft_calibrated = Column(Float, nullable=True)
    tau_medium_calibrated = Column(Float, nullable=True)
    tau_hard_calibrated = Column(Float, nullable=True)

    # ------------------------------
    # Risk thresholds (hybrid collapse)
    # ------------------------------

    risk_tau_stable = Column(Float, nullable=True)
    risk_tau_unstable = Column(Float, nullable=True)

    # ------------------------------
    # Policy / Experiment
    # ------------------------------

    policy_id = Column(Integer, nullable=False)
    task_type = Column(Text, nullable=False)

    experiment_id = Column(Integer, ForeignKey("experiments.id"), nullable=True)
    experiment = relationship("ExperimentORM", back_populates="runs")

    collapse_signals = relationship(
        "CollapseSignalORM",
        back_populates="run",
        cascade="all, delete-orphan"
    )

    # ------------------------------
    # Survival Summary (optional)
    # ------------------------------

    mean_survival_time = Column(Float, nullable=True)
    collapse_rate = Column(Float, nullable=True)
    hazard_ratio_vs_baseline = Column(Float, nullable=True)

    # ------------------------------
    # Causal summary (optional)
    # ------------------------------

    dr_ate = Column(Float, nullable=True)
    dr_ci_lower = Column(Float, nullable=True)
    dr_ci_upper = Column(Float, nullable=True)

    # ------------------------------
    # Metadata
    # ------------------------------

    start_time = Column(Float, nullable=False)
    end_time = Column(Float, nullable=True)
    status = Column(Text, default="running")

    notes = Column(Text, nullable=True)
    seed = Column(Integer, nullable=True)