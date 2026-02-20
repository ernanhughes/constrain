# constrain/data/stores/signal_discovery_store.py

from sqlalchemy.orm import sessionmaker

from constrain.data.orm.signal_discovery import SignalDiscoveryORM
from constrain.data.schemas.signal_discovery import SignalDiscoveryDTO
from constrain.data.stores.base_store import BaseSQLAlchemyStore


class SignalDiscoveryStore(BaseSQLAlchemyStore[SignalDiscoveryDTO]):
    orm_model = SignalDiscoveryORM
    default_order_by = "created_at"

    def __init__(self, sm: sessionmaker, memory=None):
        super().__init__(sm, memory)
        self.name = "signal_discovery"

    @staticmethod
    def _to_dto(row: SignalDiscoveryORM) -> SignalDiscoveryDTO:
        return SignalDiscoveryDTO.model_validate(row)
