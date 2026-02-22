# constrain/data/stores/step_store.py
from __future__ import annotations

import math
from typing import Any, List, Optional

import pandas as pd
from sqlalchemy import and_, asc, desc, func
from sqlalchemy.orm import sessionmaker

from constrain.data.orm.run import RunORM
from constrain.data.orm.step import StepORM
from constrain.data.schemas.step import StepDTO
from constrain.data.stores.base_store import BaseSQLAlchemyStore


from sqlalchemy import select

from constrain.policy.custom_types import Thresholds
from constrain.policy.threshold_query import ThresholdQuery
from constrain.policy.threshold_result import ThresholdResult


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
    

    def get_recent_unique_steps(
        self,
        limit: int = 50000,
        exclude_policy_ids: list[int] | None = None,
    ):

        def op(session):

            query = (
                session.query(StepORM)
                .join(RunORM, StepORM.run_id == RunORM.run_id)
                .order_by(desc(StepORM.timestamp))
            )

            if exclude_policy_ids:
                query = query.filter(~RunORM.policy_id.in_(exclude_policy_ids))

            rows = query.limit(limit * 3).all()  
            # Pull extra because we'll deduplicate

            data = [
                {
                    "run_id": r.run_id,
                    "problem_id": r.problem_id,
                    "iteration": r.iteration,
                    "timestamp": r.timestamp,
                    "policy_action": r.policy_action,
                    "reasoning_text": r.reasoning_text,
                    "total_energy": r.total_energy,
                    "accuracy": r.accuracy,
                    "collapse_probability": r.collapse_probability,
                    "temperature": r.temperature,
                    "gold_answer": r.gold_answer,
                    "extracted_answer": r.extracted_answer,
                }
                for r in rows
            ]

            df = pd.DataFrame(data)

            if df.empty:
                return df

            # Deduplicate by reasoning text
            df = df.drop_duplicates(subset=["reasoning_text"])

            # Keep most recent only
            df = df.sort_values("timestamp", ascending=False)

            return df.head(limit)

        return self._run(op)



    def get_energy_thresholds(self, query: ThresholdQuery) -> ThresholdResult:
        """
        Compute tau_soft/medium/hard from persisted StepORM.total_energy.

        No SQL outside the data layer.
        No recomputation.
        Uses rank-position quantiles (not tail MAX) to avoid collapse artifacts.

        Implementation note:
        We avoid window functions for SQLite portability and correctness.
        Quantiles are computed by selecting the k-th ordered energy using OFFSET.
        """

        def clamp01(x: float) -> float:
            return max(0.0, min(1.0, float(x)))

        q_soft = clamp01(query.q_soft)
        q_med = clamp01(query.q_medium)
        q_hard = clamp01(query.q_hard)

        if not (q_soft <= q_med <= q_hard):
            raise ValueError(f"Quantiles must be nondecreasing: {q_soft}, {q_med}, {q_hard}")

        def op(session):
            # ------------------------------------------------------------
            # 1) Determine which steps we will sample
            # ------------------------------------------------------------

            step_q = session.query(StepORM.id, StepORM.total_energy)

            # Join runs if we need policy filters or last_n_runs selection
            needs_run_join = (
                query.exclude_policy_ids is not None
                or query.last_n_runs is not None
                or query.include_run_ids is not None
            )
            if needs_run_join:
                step_q = step_q.join(RunORM, StepORM.run_id == RunORM.run_id)

            conditions = []

            # Scope selection
            if query.run_id is not None:
                conditions.append(StepORM.run_id == query.run_id)

            if query.include_run_ids is not None:
                if len(query.include_run_ids) == 0:
                    raise ValueError("include_run_ids cannot be empty.")
                conditions.append(StepORM.run_id.in_(list(query.include_run_ids)))

            if query.exclude_policy_ids is not None and needs_run_join:
                conditions.append(~RunORM.policy_id.in_(list(query.exclude_policy_ids)))

            if query.require_correctness_nonnull:
                conditions.append(StepORM.correctness.isnot(None))

            if conditions:
                step_q = step_q.filter(and_(*conditions))

            # last_n_runs: select most recent runs, then restrict steps to those
            if query.last_n_runs is not None:
                # choose runs by start_time desc
                run_ids_subq = (
                    session.query(RunORM.run_id)
                    .order_by(desc(RunORM.start_time))
                    .limit(int(query.last_n_runs))
                    .subquery()
                )
                step_q = step_q.filter(StepORM.run_id.in_(select(run_ids_subq.c.run_id)))

            # last_n_steps: restrict by most recent steps (timestamp desc)
            if query.last_n_steps is not None:
                ids_q = session.query(StepORM.id)

                if needs_run_join:
                    ids_q = ids_q.join(RunORM, StepORM.run_id == RunORM.run_id)
                    if query.exclude_policy_ids is not None:
                        ids_q = ids_q.filter(~RunORM.policy_id.in_(list(query.exclude_policy_ids)))

                # apply the same scope filters
                if query.run_id is not None:
                    ids_q = ids_q.filter(StepORM.run_id == query.run_id)
                if query.include_run_ids is not None:
                    ids_q = ids_q.filter(StepORM.run_id.in_(list(query.include_run_ids)))
                if query.last_n_runs is not None:
                    ids_q = ids_q.filter(StepORM.run_id.in_(select(run_ids_subq.c.run_id)))
                if query.require_correctness_nonnull:
                    ids_q = ids_q.filter(StepORM.correctness.isnot(None))

                ids_subq = (
                    ids_q.order_by(desc(StepORM.timestamp))
                    .limit(int(query.last_n_steps))
                    .subquery()
                )

                step_q = session.query(StepORM.id, StepORM.total_energy).filter(
                    StepORM.id.in_(select(ids_subq.c.id))
                )

            # Optional: ignore NULL energies (recommended)
            step_q = step_q.filter(StepORM.total_energy.isnot(None))

            # ------------------------------------------------------------
            # 2) Ensure sufficient samples
            # ------------------------------------------------------------
            n = int(step_q.count())
            if n < int(query.min_samples):
                return ThresholdResult(
                    thresholds=Thresholds(0.0, 0.0, 0.0),
                    sample_count=n,
                )

            # ------------------------------------------------------------
            # 3) Quantiles by rank (k-th ordered element)
            # ------------------------------------------------------------
            # Convert quantiles to 1-based ranks in [1..n]
            def to_rank(q: float) -> int:
                # ceil(q * n) but clamp within [1, n]
                k = int(math.ceil(q * n))
                return max(1, min(n, k))

            k_soft = to_rank(q_soft)
            k_med = to_rank(q_med)
            k_hard = to_rank(q_hard)

            # Build ids subquery once (so we don't repeat filtering logic)
            base_ids_subq = step_q.with_entities(StepORM.id).subquery()

            def energy_at_rank(k: int):
                return (
                    session.query(StepORM.total_energy)
                    .filter(StepORM.id.in_(select(base_ids_subq.c.id)))
                    .order_by(asc(StepORM.total_energy))
                    .offset(k - 1)   # k is 1-based
                    .limit(1)
                    .scalar()
                )

            tau_soft = energy_at_rank(k_soft)
            tau_med = energy_at_rank(k_med)
            tau_hard = energy_at_rank(k_hard)

            # ------------------------------------------------------------
            # 4) Defensive fallback (should be rare)
            # ------------------------------------------------------------
            if tau_soft is None or tau_med is None or tau_hard is None:
                mn, mean, mx = (
                    session.query(
                        func.min(StepORM.total_energy),
                        func.avg(StepORM.total_energy),
                        func.max(StepORM.total_energy),
                    )
                    .filter(StepORM.id.in_(select(base_ids_subq.c.id)))
                    .one()
                )
                tau_soft = mean if tau_soft is None else tau_soft
                tau_med = mean if tau_med is None else tau_med
                tau_hard = mx if tau_hard is None else tau_hard

            return ThresholdResult(
                thresholds=Thresholds(
                    float(tau_soft),
                    float(tau_med),
                    float(tau_hard),
                ),
                sample_count=n,
            )

        return self._run(op)

    def _energy_at_rank(session, base_q, k: int):
        # k is 1-based rank
        return (
            session.query(StepORM.total_energy)
            .filter(StepORM.id.in_(select(base_q.subquery().c.id)))
            .order_by(asc(StepORM.total_energy))
            .offset(max(0, k - 1))
            .limit(1)
            .scalar()
        )
