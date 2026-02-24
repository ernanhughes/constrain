# constrain/data/stores/reasoning_state_store.py

from typing import Any, List, Optional
from sqlalchemy.orm import sessionmaker

from constrain.data.orm.reasoning_state import ReasoningStateSnapshotORM
from constrain.data.schemas.reasoning_state import ReasoningStateSnapshotDTO
from constrain.data.stores.base_store import BaseSQLAlchemyStore


class ReasoningStateSnapshotStore(BaseSQLAlchemyStore):
    """
    Store for reasoning state snapshots.
    
    Enables:
    - Persist state at each step
    - Retrieve state at any point
    - Replay entire trajectories
    - Analyze state transitions
    """
    orm_model = ReasoningStateSnapshotORM
    default_order_by = ReasoningStateSnapshotORM.iteration

    def __init__(self, sm: sessionmaker, memory: Optional[Any] = None):
        super().__init__(sm, memory)
        self.name = ""

    # ─────────────────────────────────────────────────────────────
    # CREATE
    # ─────────────────────────────────────────────────────────────
    def create(self, dto: ReasoningStateSnapshotDTO) -> ReasoningStateSnapshotDTO:
        """
        Persist a state snapshot.
        """
        def op(s):
            obj = ReasoningStateSnapshotORM(**dto.to_orm_dict())
            s.add(obj)
            s.flush()
            s.refresh(obj)
            return ReasoningStateSnapshotDTO.from_orm(obj)

        return self._run(op)

    # ─────────────────────────────────────────────────────────────
    # READ BY RUN
    # ─────────────────────────────────────────────────────────────
    def get_by_run(self, run_id: str) -> List[ReasoningStateSnapshotDTO]:
        """
        Get all snapshots for a run (ordered by problem_id, iteration).
        """
        def op(s):
            objs = (
                s.query(ReasoningStateSnapshotORM)
                .filter(ReasoningStateSnapshotORM.run_id == run_id)
                .order_by(
                    ReasoningStateSnapshotORM.problem_id,
                    ReasoningStateSnapshotORM.iteration,
                )
                .all()
            )
            return [ReasoningStateSnapshotDTO.from_orm(obj) for obj in objs]

        return self._run(op)

    # ─────────────────────────────────────────────────────────────
    # READ BY PROBLEM
    # ─────────────────────────────────────────────────────────────
    def get_by_problem(
        self, run_id: str, problem_id: int
    ) -> List[ReasoningStateSnapshotDTO]:
        """
        Get full trajectory for a single problem.
        """
        def op(s):
            objs = (
                s.query(ReasoningStateSnapshotORM)
                .filter(
                    ReasoningStateSnapshotORM.run_id == run_id,
                    ReasoningStateSnapshotORM.problem_id == problem_id,
                )
                .order_by(ReasoningStateSnapshotORM.iteration)
                .all()
            )
            return [ReasoningStateSnapshotDTO.from_orm(obj) for obj in objs]

        return self._run(op)

    # ─────────────────────────────────────────────────────────────
    # READ BY STEP
    # ─────────────────────────────────────────────────────────────
    def get_by_step(self, step_id: int) -> Optional[ReasoningStateSnapshotDTO]:
        """
        Get state snapshot associated with a specific step.
        """
        def op(s):
            obj = (
                s.query(ReasoningStateSnapshotORM)
                .filter(ReasoningStateSnapshotORM.step_id == step_id)
                .first()
            )
            return ReasoningStateSnapshotDTO.from_orm(obj) if obj else None

        return self._run(op)

    # ─────────────────────────────────────────────────────────────
    # READ SPECIFIC ITERATION
    # ─────────────────────────────────────────────────────────────
    def get_at_iteration(
        self, run_id: str, problem_id: int, iteration: int
    ) -> Optional[ReasoningStateSnapshotDTO]:
        """
        Get state at a specific iteration (for replay).
        """
        def op(s):
            obj = (
                s.query(ReasoningStateSnapshotORM)
                .filter(
                    ReasoningStateSnapshotORM.run_id == run_id,
                    ReasoningStateSnapshotORM.problem_id == problem_id,
                    ReasoningStateSnapshotORM.iteration == iteration,
                )
                .first()
            )
            return ReasoningStateSnapshotDTO.from_orm(obj) if obj else None

        return self._run(op)

    # ─────────────────────────────────────────────────────────────
    # READ REVERT POINTS (For recovery analysis)
    # ─────────────────────────────────────────────────────────────
    def get_revert_points(self, run_id: str) -> List[ReasoningStateSnapshotDTO]:
        """
        Get all snapshots where a revert occurred.
        """
        def op(s):
            objs = (
                s.query(ReasoningStateSnapshotORM)
                .filter(
                    ReasoningStateSnapshotORM.run_id == run_id,
                    ReasoningStateSnapshotORM.is_after_revert == True,
                )
                .order_by(
                    ReasoningStateSnapshotORM.problem_id,
                    ReasoningStateSnapshotORM.iteration,
                )
                .all()
            )
            return [ReasoningStateSnapshotDTO.from_orm(obj) for obj in objs]

        return self._run(op)

    # ─────────────────────────────────────────────────────────────
    # REPLAY HELPERS
    # ─────────────────────────────────────────────────────────────
    def get_trajectory(self, run_id: str, problem_id: int) -> dict:
        """
        Get complete trajectory with state transitions for replay.
        
        Returns structured data for visualization/replay.
        """
        snapshots = self.get_by_problem(run_id, problem_id)
        
        if not snapshots:
            return {"problem_id": problem_id, "steps": [], "reverts": []}
        
        steps = []
        revert_points = []
        
        for snap in snapshots:
            step_data = {
                "iteration": snap.iteration,
                "current_reasoning": snap.current_reasoning,
                "temperature": snap.temperature,
                "total_energy": snap.total_energy,
                "policy_action": snap.policy_action,
                "stack_depth": snap.stack_depth,
            }
            steps.append(step_data)
            
            if snap.is_after_revert or snap.is_after_reset:
                revert_points.append({
                    "iteration": snap.iteration,
                    "type": "revert" if snap.is_after_revert else "reset",
                    "temperature_after": snap.temperature,
                })
        
        return {
            "problem_id": problem_id,
            "run_id": run_id,
            "prompt": snapshots[0].prompt_text,
            "steps": steps,
            "reverts": revert_points,
            "final_state": snapshots[-1].current_reasoning if snapshots else None,
        }