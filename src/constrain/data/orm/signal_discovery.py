# constrain/data/orm/signal_discovery.py

from sqlalchemy import Column, Float, ForeignKey, Integer, Text
from sqlalchemy.orm import relationship

from constrain.data.base import Base


class SignalDiscoveryORM(Base):
    __tablename__ = "signal_discovery"

    id = Column(Integer, primary_key=True, autoincrement=True)

    run_id = Column(Text, ForeignKey("runs.run_id"), nullable=True)

    horizon = Column(Integer, nullable=False)
    auc_score = Column(Float, nullable=False)

    feature_name = Column(Text, nullable=False)
    importance = Column(Float, nullable=False)

    created_at = Column(Float, nullable=False)

    run = relationship("RunORM")
