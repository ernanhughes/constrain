from sqlalchemy import Column, Float, ForeignKey, Integer, Text

from constrain.data.base import Base


class CausalEvaluationORM(Base):
    __tablename__ = "causal_evaluations"

    id = Column(Integer, primary_key=True, autoincrement=True)

    run_id = Column(Text, ForeignKey("runs.run_id"), nullable=False)

    method = Column(Text, nullable=False)  # "dr", "ipw", "naive"

    outcome_name = Column(Text, nullable=False)  # e.g. "collapse_next"

    ate = Column(Float, nullable=True)
    ci_lower = Column(Float, nullable=True)
    ci_upper = Column(Float, nullable=True)

    std_error = Column(Float, nullable=True)

    n_samples = Column(Integer, nullable=False)

    overlap_min = Column(Float, nullable=True)
    overlap_max = Column(Float, nullable=True)

    ess = Column(Float, nullable=True)

    treated_rate = Column(Float, nullable=True)

    created_at = Column(Float, nullable=False)