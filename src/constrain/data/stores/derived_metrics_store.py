# constrain/data/stores/derived_metrics_store.py

from __future__ import annotations

from typing import Any, Dict, List, Optional

from sqlalchemy import func
from sqlalchemy.orm import sessionmaker

from constrain.data.orm.derived_metrics import DerivedMetricsORM
from constrain.data.schemas.derived_metrics import DerivedMetricsDTO
from constrain.data.stores.base_store import BaseSQLAlchemyStore


class DerivedMetricsStore(BaseSQLAlchemyStore[DerivedMetricsDTO]):
    orm_model = DerivedMetricsORM
    default_order_by = "created_at"

    def __init__(self, sm: sessionmaker, memory: Optional[Any] = None):
        super().__init__(sm, memory)
        self.name = "derived_metrics"

    # -------------------------------------------------
    # Mapping
    # -------------------------------------------------

    @staticmethod
    def _to_dto(row: DerivedMetricsORM) -> DerivedMetricsDTO:
        return DerivedMetricsDTO.model_validate(row)

    # -------------------------------------------------
    # Upsert (Core Operation)
    # -------------------------------------------------

    def upsert(self, dto: DerivedMetricsDTO) -> DerivedMetricsDTO:
        """
        Idempotent upsert per (run_id, problem_id).
        """

        def op(s):
            row = (
                s.query(DerivedMetricsORM)
                .filter(
                    DerivedMetricsORM.run_id == dto.run_id,
                    DerivedMetricsORM.problem_id == dto.problem_id,
                )
                .first()
            )

            if row is None:
                row = DerivedMetricsORM(**dto.model_dump(exclude={"id"}))
                s.add(row)
                s.flush()
                return self._to_dto(row)

            # Update existing
            for field, value in dto.model_dump(exclude={"id"}).items():
                setattr(row, field, value)

            row.touch()
            s.flush()
            return self._to_dto(row)

        return self._run(op)

    # -------------------------------------------------
    # Retrieval
    # -------------------------------------------------

    def get_for_problem(self, run_id: str, problem_id: int) -> Optional[DerivedMetricsDTO]:
        def op(s):
            row = (
                s.query(DerivedMetricsORM)
                .filter(
                    DerivedMetricsORM.run_id == run_id,
                    DerivedMetricsORM.problem_id == problem_id,
                )
                .first()
            )
            return self._to_dto(row) if row else None

        return self._run(op)

    def list_for_run(self, run_id: str, limit: int = 10000) -> List[DerivedMetricsDTO]:
        def op(s):
            rows = (
                s.query(DerivedMetricsORM)
                .filter(DerivedMetricsORM.run_id == run_id)
                .order_by(DerivedMetricsORM.problem_id.asc())
                .limit(int(limit))
                .all()
            )
            return [self._to_dto(r) for r in rows]

        return self._run(op)

    # -------------------------------------------------
    # Research Aggregates
    # -------------------------------------------------

    def collapse_rate(self, run_id: str) -> float:
        """
        % of problems that collapsed.
        """

        def op(s):
            total = (
                s.query(func.count(DerivedMetricsORM.id))
                .filter(DerivedMetricsORM.run_id == run_id)
                .scalar()
                or 0
            )

            if total == 0:
                return 0.0

            collapsed = (
                s.query(func.count(DerivedMetricsORM.id))
                .filter(
                    DerivedMetricsORM.run_id == run_id,
                    DerivedMetricsORM.collapse_detected == True,
                )
                .scalar()
                or 0
            )

            return float(collapsed) / float(total)

        return self._run(op)

    def recovery_rate(self, run_id: str) -> float:
        """
        % of collapsed cases that recovered.
        """

        def op(s):
            collapsed = (
                s.query(func.count(DerivedMetricsORM.id))
                .filter(
                    DerivedMetricsORM.run_id == run_id,
                    DerivedMetricsORM.collapse_detected == True,
                )
                .scalar()
                or 0
            )

            if collapsed == 0:
                return 0.0

            recovered = (
                s.query(func.count(DerivedMetricsORM.id))
                .filter(
                    DerivedMetricsORM.run_id == run_id,
                    DerivedMetricsORM.recovered == True,
                )
                .scalar()
                or 0
            )

            return float(recovered) / float(collapsed)

        return self._run(op)

    def mean_drift_slope(self, run_id: str) -> float:
        """
        Average drift slope across problems.
        """

        def op(s):
            val = (
                s.query(func.avg(DerivedMetricsORM.drift_slope))
                .filter(DerivedMetricsORM.run_id == run_id)
                .scalar()
            )
            return float(val or 0.0)

        return self._run(op)

    def summary_stats(self, run_id: str) -> Dict[str, float]:
        """
        Paper-ready summary statistics.
        """

        def op(s):
            total = (
                s.query(func.count(DerivedMetricsORM.id))
                .filter(DerivedMetricsORM.run_id == run_id)
                .scalar()
                or 0
            )

            collapse_rate = self.collapse_rate(run_id)
            recovery_rate = self.recovery_rate(run_id)
            mean_slope = self.mean_drift_slope(run_id)

            return {
                "total_problems": total,
                "collapse_rate": collapse_rate,
                "recovery_rate": recovery_rate,
                "mean_drift_slope": mean_slope,
            }

        return self._run(op)
