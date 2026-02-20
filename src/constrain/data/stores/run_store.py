# constrain/data/stores/run_store.py

from __future__ import annotations

from typing import Any, Optional

from sqlalchemy.orm import sessionmaker

from constrain.data.orm.run import RunORM
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
