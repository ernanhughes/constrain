# constrain/data/stores/stage2_evaluation_store.py

from __future__ import annotations

from typing import Any, Optional, List

from sqlalchemy.orm import sessionmaker
from sqlalchemy import select

from constrain.data.orm.policy_evaluation import PolicyEvaluationORM
from constrain.data.schemas.policy_evaluation import PolicyEvaluationDTO
from constrain.data.stores.base_store import BaseSQLAlchemyStore


class PolicyEvaluationStore(BaseSQLAlchemyStore[PolicyEvaluationDTO]):
    orm_model = PolicyEvaluationORM
    default_order_by = "created_at"

    def __init__(self, sm: sessionmaker, memory: Optional[Any] = None):
        super().__init__(sm, memory)
        self.name = "policy_evaluations"

    @staticmethod
    def _to_dto(row: PolicyEvaluationORM) -> PolicyEvaluationDTO:
        return PolicyEvaluationDTO.model_validate(row)

    @staticmethod
    def _from_dto(dto: PolicyEvaluationDTO) -> PolicyEvaluationORM:
        return PolicyEvaluationORM(**dto.model_dump(exclude={"id"}))

    # ------------------------------------------------------------
    # Create single evaluation
    # ------------------------------------------------------------

    def create(self, dto: PolicyEvaluationDTO) -> PolicyEvaluationDTO:
        def op(s):
            obj = PolicyEvaluationORM(**dto.model_dump(exclude={"id"}))
            s.add(obj)
            s.flush()
            return self._to_dto(obj)

        return self._run(op)

    # ------------------------------------------------------------
    # Bulk create evaluations
    # ------------------------------------------------------------

    def bulk_create(self, dtos: List[PolicyEvaluationDTO]) -> int:
        def op(s):
            objects = [self._from_dto(dto) for dto in dtos]
            s.add_all(objects)
            s.flush()
            return len(objects)

        return self._run(op)

    # ------------------------------------------------------------
    # Get by run_id
    # ------------------------------------------------------------

    def get_by_run_id(self, run_id: str) -> List[PolicyEvaluationDTO]:
        def op(s):
            stmt = select(self.orm_model).where(
                self.orm_model.run_id == run_id
            ).order_by(self.orm_model.created_at.asc())
            rows = s.execute(stmt).scalars().all()
            return [self._to_dto(row) for row in rows]

        return self._run(op)

    # ------------------------------------------------------------
    # Get by policy_id (cross-run)
    # ------------------------------------------------------------

    def get_by_policy_id(
        self, policy_id: int, experiment_id: Optional[int] = None
    ) -> List[PolicyEvaluationDTO]:
        def op(s):
            stmt = select(PolicyEvaluationORM).where(
                PolicyEvaluationORM.policy_id == policy_id
            )
            if experiment_id is not None:
                stmt = stmt.where(PolicyEvaluationORM.experiment_id == experiment_id)
            rows = s.execute(stmt).scalars().all()
            return [self._to_dto(row) for row in rows]

        return self._run(op)

    # ------------------------------------------------------------
    # Get comparison data (multiple policies)
    # ------------------------------------------------------------

    def get_comparison_data(
        self, policy_ids: List[int], experiment_id: Optional[int] = None
    ) -> List[PolicyEvaluationDTO]:
        def op(s):
            stmt = select(self.orm_model).where(
                self.orm_model.policy_id.in_(policy_ids)
            )
            if experiment_id is not None:
                stmt = stmt.where(self.orm_model.experiment_id == experiment_id)
            rows = s.execute(stmt).scalars().all()
            return [self._to_dto(row) for row in rows]

        return self._run(op)

    # ------------------------------------------------------------
    # Get by problem (for debugging)
    # ------------------------------------------------------------

    def get_by_problem(
        self, run_id: str, problem_id: int
    ) -> Optional[PolicyEvaluationDTO]:
        def op(s):
            stmt = select(self.orm_model).where(
                self.orm_model.run_id == run_id,
                self.orm_model.problem_id == problem_id,
            )
            row = s.execute(stmt).scalars().first()
            return self._to_dto(row) if row else None

        return self._run(op)