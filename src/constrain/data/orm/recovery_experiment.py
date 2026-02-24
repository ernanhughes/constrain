from __future__ import annotations

from sqlalchemy import Column, Float, Integer, Text, Boolean

from constrain.data.base import Base


class RecoveryExperimentORM(Base):
    __tablename__ = "recovery_experiments"

    id = Column(Integer, primary_key=True, autoincrement=True)

    # Experiment metadata
    experiment_name = Column(Text, nullable=False)
    experiment_type = Column(Text, nullable=False)  # "intervention_recovery" | "energy_delta" | "timing_analysis"

    # Scope configuration
    run_ids = Column(Text, nullable=False)  # JSON list of run IDs analyzed
    problem_filter = Column(Text, nullable=True)  # Optional JSON filter criteria

    # Analysis configuration
    energy_threshold = Column(Float, nullable=True)
    accuracy_delta_threshold = Column(Float, nullable=True)
    min_pre_intervention_steps = Column(Integer, nullable=True)
    min_post_intervention_steps = Column(Integer, nullable=True)

    # Timing
    start_time = Column(Float, nullable=False)
    end_time = Column(Float, nullable=True)
    status = Column(Text, nullable=False)  # "running" | "completed" | "failed"

    # Results (stored as JSON strings)
    summary_metrics = Column(Text, nullable=True)  # {mean_recovery, std_recovery, n_interventions, etc.}
    per_problem_results = Column(Text, nullable=True)  # Optional detailed results per problem
    statistical_tests = Column(Text, nullable=True)  # {p_value, ci_lower, ci_upper, effect_size}

    # Diagnostics
    overlap_warning = Column(Boolean, default=False)
    confounding_warning = Column(Boolean, default=False)
    notes = Column(Text, nullable=True)

