# constrain/data/stores/step_store.py

from __future__ import annotations

from typing import Any, List, Optional

from sqlalchemy import and_
from sqlalchemy.orm import sessionmaker
from sqlalchemy import func

from constrain.data.orm.step import StepORM
from constrain.data.schemas.step import StepDTO
from constrain.data.stores.base_store import BaseSQLAlchemyStore


class StepStore(BaseSQLAlchemyStore[StepDTO]):
    orm_model = StepORM
    default_order_by = "timestamp"

    def __init__(self, sm: sessionmaker, memory: Optional[Any] = None):
        super().__init__(sm, memory)
        self.name = "steps"

    @staticmethod
    def _to_dto(row: StepORM) -> StepDTO:
        return StepDTO.model_validate(row)

    def create(self, dto: StepDTO) -> StepDTO:
        def op(s):
            obj = StepORM(**dto.model_dump(exclude={"id"}))
            s.add(obj)
            s.flush()
            return self._to_dto(obj)

        return self._run(op)

    def filter(
        self,
        *,
        run_id: Optional[str] = None,
        problem_id: Optional[int] = None,
        iteration: Optional[int] = None,
        min_iteration: Optional[int] = None,
        max_iteration: Optional[int] = None,
        limit: Optional[int] = None,
    ) -> List[StepDTO]:
        """
        Flexible filtering for steps.

        Examples:
            memory.steps.filter(run_id="abc")
            memory.steps.filter(run_id="abc", problem_id=3)
            memory.steps.filter(run_id="abc", min_iteration=10)
        """

        def op(session):
            query = session.query(StepORM)

            conditions = []

            if run_id is not None:
                conditions.append(StepORM.run_id == run_id)

            if problem_id is not None:
                conditions.append(StepORM.problem_id == problem_id)

            if iteration is not None:
                conditions.append(StepORM.iteration == iteration)

            if min_iteration is not None:
                conditions.append(StepORM.iteration >= min_iteration)

            if max_iteration is not None:
                conditions.append(StepORM.iteration <= max_iteration)

            if conditions:
                query = query.filter(and_(*conditions))

            query = query.order_by(StepORM.iteration.asc())

            if limit is not None:
                query = query.limit(limit)

            rows = query.all()

            return [
                StepDTO.model_validate(row)
                for row in rows
            ]

        return self._run(op)

    def get_by_run(self, run_id: str) -> List[StepDTO]:

        def op(session):
            rows = (
                session.query(StepORM)
                .filter(StepORM.run_id == run_id)
                .order_by(StepORM.problem_id.asc(), StepORM.iteration.asc())
                .all()
            )

            return [StepDTO.model_validate(r) for r in rows]

        return self._run(op)

    
    def get_reasoning_by_prompt(self, prompt_text: str, temperature: float):
        def op(s):
            return (
                s.query(StepORM)
                .filter(
                    StepORM.prompt_text == prompt_text
                )
                .order_by(func.random())
                .first()
            )
        return self._run(op)


    def get_distinct_prompts(self, limit: int | None = None):
        def op(s):
            q = s.query(self.orm_model.prompt_text).distinct()
            if limit:
                q = q.limit(limit)
            return [row[0] for row in q.all()]
        return self._run(op)