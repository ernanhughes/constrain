from sqlalchemy import Column, Float, ForeignKey, Integer, Text
from sqlalchemy.orm import relationship

from constrain.data.base import Base


class StepORM(Base):
    __tablename__ = "steps"

    id = Column(Integer, primary_key=True, autoincrement=True)

    run_id = Column(Text, ForeignKey("runs.run_id"), nullable=False)
    problem_id = Column(Integer, nullable=False)
    iteration = Column(Integer, nullable=False)

    prompt_text = Column(Text, nullable=False)
    reasoning_text = Column(Text, nullable=False)

    gold_answer = Column(Text, nullable=True)
    extracted_answer = Column(Text, nullable=True)

    total_energy = Column(Float, nullable=False)
    grounding_energy = Column(Float, nullable=False)
    stability_energy = Column(Float, nullable=False)

    accuracy = Column(Float, nullable=True)
    correctness = Column(Integer, nullable=True)

    temperature = Column(Float, nullable=False)
    policy_action = Column(Text, nullable=False)
    phase = Column(Text, nullable=False)

    timestamp = Column(Float, nullable=False)

    run = relationship("RunORM")
