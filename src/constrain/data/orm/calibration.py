from __future__ import annotations

from sqlalchemy import Column, Float, ForeignKey, Index, Integer, Text
from sqlalchemy.orm import relationship

from constrain.data.base import Base


class CalibrationORM(Base):
    __tablename__ = "calibrations"

    id = Column(Integer, primary_key=True, autoincrement=True)

    run_id = Column(Text, ForeignKey("runs.run_id"), nullable=False)

    policy_mode = Column(Text, nullable=False)  # static | recursive | adaptive | dynamic

    tau_soft = Column(Float, nullable=False)
    tau_medium = Column(Float, nullable=False)
    tau_hard = Column(Float, nullable=False)

    sample_count = Column(Integer, nullable=True)

    created_at = Column(Float, nullable=False)

    run = relationship("RunORM", backref="calibrations")

    __table_args__ = (
        Index("idx_calibration_run", "run_id"),
    )
