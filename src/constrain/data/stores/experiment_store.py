# constrain/data/stores/experiment_store.py

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import sessionmaker

from constrain.data.orm.experiment import ExperimentORM
from constrain.data.schemas.experiment import ExperimentDTO
from constrain.data.stores.base_store import BaseSQLAlchemyStore


class ExperimentStore(BaseSQLAlchemyStore[ExperimentDTO]):
    orm_model = ExperimentORM
    default_order_by = "start_time"

    def __init__(self, sm: sessionmaker, memory: Optional[Any] = None):
        super().__init__(sm, memory)
        self.name = "experiments"

    @staticmethod
    def _to_dto(row: ExperimentORM) -> ExperimentDTO:
        return ExperimentDTO.from_orm(row)

    def create(self, dto: ExperimentDTO) -> ExperimentDTO:
        def op(s):
            obj = ExperimentORM(
                experiment_name=dto.experiment_name,
                experiment_type=dto.experiment_type,
                policy_ids=json.dumps(dto.policy_ids),
                seeds=json.dumps(dto.seeds),
                num_problems=dto.num_problems,
                num_recursions=dto.num_recursions,
                start_time=dto.start_time,
                end_time=dto.end_time,
                status=dto.status,
                results_summary=json.dumps(dto.results_summary) if dto.results_summary else None,
                notes=dto.notes,
                git_commit=dto.git_commit,
            )
            s.add(obj)
            s.flush()
            return self._to_dto(obj)

        return self._run(op)

    def update(self, experiment_id: int, updates: Dict[str, Any]) -> Optional[ExperimentDTO]:
        def op(s):
            obj = s.query(ExperimentORM).filter_by(id=experiment_id).first()
            if obj is None:
                return None

            for key, value in updates.items():
                if key in ["policy_ids", "seeds", "results_summary"] and value is not None:
                    value = json.dumps(value)
                if hasattr(obj, key):
                    setattr(obj, key, value)

            s.flush()
            return self._to_dto(obj)

        return self._run(op)

    def get_by_id(self, experiment_id: int) -> Optional[ExperimentDTO]:
        def op(s):
            row = s.query(ExperimentORM).filter_by(id=experiment_id).first()
            return self._to_dto(row) if row else None

        return self._run(op)

    def get_all(self, limit: Optional[int] = None) -> List[ExperimentDTO]:
        def op(s):
            query = s.query(ExperimentORM).order_by(ExperimentORM.start_time.desc())
            if limit:
                query = query.limit(limit)
            return [self._to_dto(row) for row in query.all()]

        return self._run(op)

    def get_by_status(self, status: str) -> List[ExperimentDTO]:
        def op(s):
            rows = s.query(ExperimentORM).filter_by(status=status).all()
            return [self._to_dto(row) for row in rows]

        return self._run(op)

    def get_running(self) -> List[ExperimentDTO]:
        return self.get_by_status("running")

    def get_completed(self, limit: Optional[int] = None) -> List[ExperimentDTO]:
        def op(s):
            query = s.query(ExperimentORM).filter_by(status="completed").order_by(
                ExperimentORM.start_time.desc()
            )
            if limit:
                query = query.limit(limit)
            return [self._to_dto(row) for row in query.all()]

        return self._run(op)