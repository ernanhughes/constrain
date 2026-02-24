"""
ORM model for reasoning state snapshots.
"""
from sqlalchemy import Column, Float, Integer, Text, Boolean, ForeignKey
from constrain.data.base import Base


class ReasoningStateSnapshotORM(Base):
    __tablename__ = "reasoning_state_snapshots"

    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(Text, nullable=False, index=True)
    problem_id = Column(Integer, nullable=False, index=True)
    step_id = Column(Integer, ForeignKey("steps.id"), nullable=True, index=True)
    
    # ─────────────────────────────────────────────────────────────
    # EXISTING FIELDS (keep these)
    # ─────────────────────────────────────────────────────────────
    iteration = Column(Integer, nullable=False)  # ← Now means ACCEPTED DEPTH
    prompt_text = Column(Text, nullable=False)
    current_reasoning = Column(Text, nullable=False)
    history_json = Column(Text, nullable=False)
    temperature = Column(Float, nullable=False)
    initial_temperature = Column(Float, nullable=False)
    stack_depth = Column(Integer, nullable=False)
    is_after_revert = Column(Boolean, default=False)
    is_after_reset = Column(Boolean, default=False)
    parent_snapshot_id = Column(Integer, ForeignKey("reasoning_state_snapshots.id"), nullable=True)
    total_energy = Column(Float, nullable=True)
    grounding_energy = Column(Float, nullable=True)
    stability_energy = Column(Float, nullable=True)
    policy_action = Column(Text, nullable=True)
    policy_action_reason = Column(Text, nullable=True)
    created_at = Column(Float, nullable=False)
    
    attempt = Column(Integer, nullable=False, default=0)  # Model call count
    energy_slope = Column(Float, nullable=True)
    violation_level = Column(Text, nullable=True)  # none/soft/medium/hard
    consecutive_hard = Column(Integer, nullable=True, default=0)
    consecutive_rising = Column(Integer, nullable=True, default=0)
    collapse_flag = Column(Boolean, default=False)