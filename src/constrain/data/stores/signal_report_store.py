from typing import Any, Optional

from sqlalchemy.orm import sessionmaker

from constrain.data.orm.signal_report import SignalReportORM
from constrain.data.schemas.signal_report import SignalReportDTO
from constrain.data.stores.base_store import BaseSQLAlchemyStore


class SignalReportStore(BaseSQLAlchemyStore):
    orm_model = SignalReportORM
    default_order_by = SignalReportORM.created_at


    def __init__(self, sm: sessionmaker, memory: Optional[Any] = None):
        super().__init__(sm, memory)
        self.name = "signal_reports"

    def create(self, dto: SignalReportDTO) -> SignalReportDTO:
        def op(s):
            obj = SignalReportORM(**dto.model_dump(exclude={"id"}))
            s.add(obj)
            s.flush()
            s.refresh(obj)
            return SignalReportDTO.model_validate(obj)

        return self._run(op)
