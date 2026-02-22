# constrain/data/stores/policy_event_store.py

from __future__ import annotations

import time
from typing import Optional

from sqlalchemy.orm import sessionmaker

from constrain.data.orm.policy_event import PolicyEventORM
from constrain.data.schemas.policy_event import PolicyEventDTO
from constrain.data.stores.base_store import BaseSQLAlchemyStore


class PolicyEventStore(BaseSQLAlchemyStore[PolicyEventDTO]):
    orm_model = PolicyEventORM
    default_order_by = "created_at"

    def __init__(self, sm: sessionmaker, memory: Optional[object] = None):
        super().__init__(sm, memory)
        self.name = "policy_events"

    @staticmethod
    def _to_dto(row: PolicyEventORM) -> PolicyEventDTO:
        return PolicyEventDTO.model_validate(row)

    # ------------------------------------------------------------
    # Create event
    # ------------------------------------------------------------

    def create_event(
        self,
        *,
        run_id: str,
        step_id: int,
        policy_id: int,
        tau_soft: float,
        tau_medium: float,
        tau_hard: float,
        action: str,
        collapse_probability: Optional[float] = None,
    ) -> PolicyEventDTO:

        now = time.time()

        def op(s):
            obj = PolicyEventORM(
                run_id=run_id,
                step_id=step_id,
                policy_id=policy_id,
                tau_soft=float(tau_soft),
                tau_medium=float(tau_medium),
                tau_hard=float(tau_hard),
                action=action,
                collapse_probability=collapse_probability,
                created_at=now,
            )
            s.add(obj)
            s.flush()
            return obj

        row = self._run(op)
        return self._to_dto(row)