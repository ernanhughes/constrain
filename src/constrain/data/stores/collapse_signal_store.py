# constrain/data/stores/collapse_signal_store.py
from __future__ import annotations

from typing import Any, List, Optional

from sqlalchemy import select
from sqlalchemy.orm import sessionmaker

from constrain.data.orm.collapse_signal import CollapseSignalORM
from constrain.data.schemas.collapse_signal import CollapseSignalDTO
from constrain.data.stores.base_store import BaseSQLAlchemyStore


class CollapseSignalStore(BaseSQLAlchemyStore[CollapseSignalDTO]):
    orm_model = CollapseSignalORM
    default_order_by = "created_at"

    def __init__(self, sm: sessionmaker, memory: Optional[Any] = None):
        super().__init__(sm, memory)
        self.name = "collapse_signals"

    @staticmethod
    def _to_dto(row: CollapseSignalORM) -> CollapseSignalDTO:
        return CollapseSignalDTO.model_validate(row)

    @staticmethod
    def _from_dto(dto: CollapseSignalDTO) -> CollapseSignalORM:
        data = dto.model_dump()
        return CollapseSignalORM(**{k: v for k, v in data.items() if k != "id"})

    def create(self, dto: CollapseSignalDTO) -> CollapseSignalDTO:
        def op(s):
            obj = self._from_dto(dto)
            s.add(obj)
            s.flush()
            return self._to_dto(obj)
        return self._run(op)

    def get_by_run_id(self, run_id: str) -> List[CollapseSignalDTO]:
        def op(s):
            stmt = select(self.orm_model).where(self.orm_model.run_id == run_id)
            rows = s.execute(stmt).scalars().all()
            return [self._to_dto(row) for row in rows]
        return self._run(op)

    def get_by_experiment_id(self, experiment_id: int) -> List[CollapseSignalDTO]:
        def op(s):
            stmt = select(self.orm_model).where(self.orm_model.experiment_id == experiment_id)
            rows = s.execute(stmt).scalars().all()
            return [self._to_dto(row) for row in rows]
        return self._run(op)