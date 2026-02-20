from __future__ import annotations

import time
from typing import Any, Optional

from sqlalchemy.orm import sessionmaker

from constrain.data.orm.calibration import CalibrationORM
from constrain.data.schemas.calibration import CalibrationDTO
from constrain.data.stores.base_store import BaseSQLAlchemyStore


class CalibrationStore(BaseSQLAlchemyStore[CalibrationDTO]):
    orm_model = CalibrationORM
    default_order_by = "created_at"

    def __init__(self, sm: sessionmaker, memory: Optional[Any] = None):
        super().__init__(sm, memory)
        self.name = "calibrations"

    # -------------------------------------------------
    # DTO conversion
    # -------------------------------------------------

    @staticmethod
    def _to_dto(row: CalibrationORM) -> CalibrationDTO:
        return CalibrationDTO(
            id=row.id,
            run_id=row.run_id,
            policy_mode=row.policy_mode,
            tau_soft=row.tau_soft,
            tau_medium=row.tau_medium,
            tau_hard=row.tau_hard,
            sample_count=row.sample_count,
            created_at=row.created_at,
        )

    # -------------------------------------------------
    # Create Calibration
    # -------------------------------------------------

    def create_calibration(
        self,
        *,
        run_id: str,
        policy_mode: str,
        tau_soft: float,
        tau_medium: float,
        tau_hard: float,
        sample_count: Optional[int] = None,
    ) -> CalibrationDTO:

        now = time.time()

        def op(s):
            obj = CalibrationORM(
                run_id=run_id,
                policy_mode=policy_mode,
                tau_soft=float(tau_soft),
                tau_medium=float(tau_medium),
                tau_hard=float(tau_hard),
                sample_count=sample_count,
                created_at=now,
            )
            s.add(obj)
            s.flush()
            return obj

        row = self._run(op)
        return self._to_dto(row)

    # -------------------------------------------------
    # Get Latest Calibration
    # -------------------------------------------------

    def get_latest(
        self,
        *,
        policy_mode: str,
        run_id: Optional[str] = None,
    ) -> Optional[CalibrationDTO]:

        def op(s):
            q = s.query(CalibrationORM).filter(
                CalibrationORM.policy_mode == policy_mode
            )

            if run_id:
                q = q.filter(CalibrationORM.run_id == run_id)

            row = q.order_by(CalibrationORM.created_at.desc()).first()
            return row

        row = self._run(op)

        if not row:
            return None

        return self._to_dto(row)
