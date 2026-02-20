# constrain/data/stores/intervention_store.py

from __future__ import annotations

from typing import Any, Optional

from sqlalchemy.orm import sessionmaker

from constrain.data.orm.intervention import InterventionORM
from constrain.data.schemas.intervention import InterventionDTO
from constrain.data.stores.base_store import BaseSQLAlchemyStore


class InterventionStore(BaseSQLAlchemyStore[InterventionDTO]):
    orm_model = InterventionORM
    default_order_by = "timestamp"

    def __init__(self, sm: sessionmaker, memory: Optional[Any] = None):
        super().__init__(sm, memory)
        self.name = "interventions"

    @staticmethod
    def _to_dto(row: InterventionORM) -> InterventionDTO:
        return InterventionDTO.model_validate(row)

    def create(self, dto: InterventionDTO) -> InterventionDTO:
        def op(s):
            obj = InterventionORM(**dto.model_dump(exclude={"id"}))
            s.add(obj)
            s.flush()
            return self._to_dto(obj)

        return self._run(op)
