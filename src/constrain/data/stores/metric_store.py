from __future__ import annotations

import time
from typing import Dict

from sqlalchemy.orm import sessionmaker

from constrain.data.orm.metric import MetricORM
from constrain.data.schemas.metric import MetricDTO
from constrain.data.stores.base_store import BaseSQLAlchemyStore


class MetricStore(BaseSQLAlchemyStore[MetricDTO]):
    orm_model = MetricORM
    default_order_by = "created_at"

    def __init__(self, sm: sessionmaker, memory=None):
        super().__init__(sm, memory=memory)
        self.name = "metrics"

    @staticmethod
    def _to_dto(row: MetricORM) -> MetricDTO:
        return MetricDTO.model_validate(row)

    def create(self, dto: MetricDTO) -> MetricDTO:
        def op(s):
            obj = MetricORM(**dto.model_dump(exclude={"id"}))
            s.add(obj)
            s.flush()
            return self._to_dto(obj)

        return self._run(op)

    def bulk_from_dict(
        self,
        *,
        step_id: int,
        stage: str,
        metrics: Dict[str, float],
    ):
        now = time.time()

        def op(s):
            objs = []
            for name, value in metrics.items():
                objs.append(
                    MetricORM(
                        step_id=step_id,
                        stage=stage,
                        metric_name=name,
                        metric_value=float(value),
                        created_at=now,
                    )
                )
            s.add_all(objs)
            s.flush()
            return len(objs)

        return self._run(op)
