from constrain.data.orm.signal_report import SignalReportORM
from constrain.data.stores.base_store import BaseSQLAlchemyStore


class SignalReportStore(BaseSQLAlchemyStore):
    orm_model = SignalReportORM
    default_order_by = SignalReportORM.created_at
