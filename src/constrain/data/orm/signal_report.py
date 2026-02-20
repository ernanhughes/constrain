from sqlalchemy import Column, Float, Integer, LargeBinary, Text
from sqlalchemy.orm import relationship

from constrain.data.base import Base


class SignalReportORM(Base):
    __tablename__ = "signal_reports"

    id = Column(Integer, primary_key=True, autoincrement=True)

    run_id = Column(Text, nullable=False)

    # Core metrics
    mean_energy = Column(Float, nullable=True)
    mean_energy_slope = Column(Float, nullable=True)
    intervention_recovery_delta = Column(Float, nullable=True)

    # Predictive model
    auc = Column(Float, nullable=True)

    # Serialized feature importance JSON
    feature_importance_json = Column(Text, nullable=True)

    created_at = Column(Float, nullable=False)
