from sqlalchemy import Column, Float, ForeignKey, Integer, Text
from sqlalchemy.orm import relationship

from constrain.data.base import Base


class InterventionORM(Base):
    __tablename__ = "interventions"

    id = Column(Integer, primary_key=True, autoincrement=True)

    run_id = Column(Text, ForeignKey("runs.run_id"), nullable=False)
    problem_id = Column(Integer, nullable=False)
    iteration = Column(Integer, nullable=False)

    threshold = Column(Text, nullable=False)
    rationale = Column(Text, nullable=True)
    reverted_to = Column(Integer, nullable=True)

    new_temperature = Column(Float, nullable=True)
    timestamp = Column(Float, nullable=False)

    run = relationship("RunORM")
