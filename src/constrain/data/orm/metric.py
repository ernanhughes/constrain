from sqlalchemy import Column, Float, ForeignKey, Integer, Text
from sqlalchemy.orm import relationship

from constrain.data.base import Base


class MetricORM(Base):
    __tablename__ = "metrics"

    id = Column(Integer, primary_key=True, autoincrement=True)

    step_id = Column(Integer, ForeignKey("steps.id"), nullable=False)

    stage = Column(Text, nullable=False)
    metric_name = Column(Text, nullable=False)
    metric_value = Column(Float, nullable=False)

    created_at = Column(Float, nullable=False)

    step = relationship("StepORM")
