
from __future__ import annotations

from typing import Any, Dict, Optional

from sqlalchemy import desc
from sqlalchemy.orm import sessionmaker

from constrain.data.orm.run import RunORM
from constrain.data.orm.step import StepORM
from constrain.data.schemas.run import RunDTO
from constrain.data.stores.base_store import BaseSQLAlchemyStore


class RunStore(BaseSQLAlchemyStore[RunDTO]):
    orm_model = RunORM
    default_order_by = "start_time"

    def __init__(self, sm: sessionmaker, memory: Optional[Any] = None):
        super().__init__(sm, memory)
        self.name = "runs"

    @staticmethod
    def _to_dto(row: RunORM) -> RunDTO:
        return RunDTO.model_validate(row)

    def create(self, dto: RunDTO) -> RunDTO:
        def op(s):
            obj = RunORM(**dto.model_dump())
            s.add(obj)
            s.flush()
            return self._to_dto(obj)

        return self._run(op)

    def get_previous_completed_run(self, current_run_id: str) -> Optional[str]:
        """
        Returns the most recent completed run_id that is NOT the current run.
        Returns None if none exists.
        """

        def op(s):
            row = (
                s.query(RunORM)
                .filter(RunORM.status == "completed")
                .filter(RunORM.run_id != current_run_id)
                .order_by(RunORM.start_time.desc())
                .first()
            )

            return row.run_id if row else None

        return self._run(op)
    

    def update(self, run_id: str, updates: Dict[str, Any]) -> Optional[RunDTO]:
        def op(s):
            obj = s.query(self.orm_model).filter_by(run_id=run_id).first()
            if obj is None:
                return None
            for key, value in updates.items():
                if hasattr(obj, key):
                    setattr(obj, key, value)
            s.flush()
            return self._to_dto(obj)
        return self._run(op)
    

    def get_all(self, desc_start_time: bool = False) -> list[RunDTO]:
        def op(s):
            query = s.query(self.orm_model)

            if desc_start_time:
                query = query.order_by(desc(RunORM.start_time))
            else:
                query = query.order_by(RunORM.start_time)

            rows = query.all()
            return [self._to_dto(row) for row in rows]

        return self._run(op)


    def get_recent_runs(self, limit: int = 50) -> list[RunDTO]:
        def op(s):
            rows = (
                s.query(RunORM)
                .join(StepORM, StepORM.run_id == RunORM.run_id)
                .group_by(RunORM.run_id)
                .order_by(RunORM.start_time.desc())
                .limit(limit)
                .all()
            )
            return [self._to_dto(row) for row in rows]

        return self._run(op)