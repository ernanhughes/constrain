from sqlalchemy import Column, Float, ForeignKey, Integer, Text
from sqlalchemy.orm import relationship

from constrain.data.base import Base


class CollapseSignalORM(Base):
    """
    Persisted results from collapse prediction / signal discovery analysis.
    
    Stores feature importance, AUC metrics, and diagnostics for runs
    where predictive signals for intervention-triggered collapse were analyzed.
    """
    __tablename__ = "collapse_signals"

    # Primary key
    id = Column(Integer, primary_key=True, autoincrement=True)

    # Link to source run (required)
    run_id = Column(Text, ForeignKey("runs.run_id"), nullable=False)

    # Optional link to experiment grouping
    experiment_id = Column(Integer, ForeignKey("experiments.id"), nullable=True)

    # Analysis configuration
    prediction_horizon = Column(Integer, nullable=False)  # t+horizon for collapse label
    n_samples = Column(Integer, nullable=False)  # valid samples used for training

    # Model performance metrics
    auc_mean = Column(Float, nullable=True)  # mean AUC across CV folds
    auc_std = Column(Float, nullable=True)   # std of AUC across folds

    # Feature importance (JSON-encoded list of {feature, importance} dicts)
    feature_importance_json = Column(Text, nullable=True)

    # Diagnostic metrics
    mean_energy = Column(Float, nullable=True)  # avg total_energy in dataset
    intervention_energy_delta = Column(Float, nullable=True)  # energy change post-intervention

    # Metadata
    created_at = Column(Float, nullable=False)

    # Relationships
    run = relationship("RunORM", back_populates="collapse_signals")
    experiment = relationship("ExperimentORM", back_populates="collapse_signals")