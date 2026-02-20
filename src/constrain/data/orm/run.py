from sqlalchemy import Column, Float, Integer, Text
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

    tau_soft = Column(Float, nullable=False)
    tau_medium = Column(Float, nullable=False)
    tau_hard = Column(Float, nullable=False)

    policy_id = Column(Integer, nullable=False)
    task_type = Column(Text, nullable=False)

    start_time = Column(Float, nullable=False)
    end_time = Column(Float, nullable=True)
    status = Column(Text, default="running")

    notes = Column(Text, nullable=True)

    tau_soft_calibrated = Column(Float, nullable=True)
    tau_medium_calibrated = Column(Float, nullable=True)
    tau_hard_calibrated = Column(Float, nullable=True)
