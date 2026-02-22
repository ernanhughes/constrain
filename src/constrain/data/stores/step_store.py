# constrain/data/stores/step_store.py
from __future__ import annotations

from typing import Any, List, Optional

import pandas as pd
from sqlalchemy import and_, desc, func, Integer
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

            # last_n_steps: restrict by most recent steps
            if query.last_n_steps is not None:
                # easiest: order by timestamp desc, limit, then compute thresholds from that subset
                # We do this by turning it into a subquery of ids.
                ids_subq = (
                    session.query(StepORM.id)
                    .filter(and_(*conditions)) if conditions else session.query(StepORM.id)
                )

                if needs_run_join:
                    ids_subq = ids_subq.join(RunORM, StepORM.run_id == RunORM.run_id)
                    if query.exclude_policy_ids is not None:
                        ids_subq = ids_subq.filter(~RunORM.policy_id.in_(list(query.exclude_policy_ids)))

                if query.run_id is not None:
                    ids_subq = ids_subq.filter(StepORM.run_id == query.run_id)
                if query.include_run_ids is not None:
                    ids_subq = ids_subq.filter(StepORM.run_id.in_(list(query.include_run_ids)))
                if query.last_n_runs is not None:
                    ids_subq = ids_subq.filter(StepORM.run_id.in_(select(run_ids_subq.c.run_id)))
                if query.require_correctness_nonnull:
                    ids_subq = ids_subq.filter(StepORM.correctness.isnot(None))

                ids_subq = (
                    ids_subq.order_by(desc(StepORM.timestamp))
                    .limit(int(query.last_n_steps))
                    .subquery()
                )

                step_q = session.query(StepORM.id, StepORM.total_energy).filter(StepORM.id.in_(select(ids_subq.c.id)))

            # ------------------------------------------------------------
            # 2) Ensure sufficient samples
            # ------------------------------------------------------------
            n = step_q.count()
            if n < int(query.min_samples):
                raise ValueError(f"Not enough samples for thresholds: n={n} < min_samples={query.min_samples}")

            # ------------------------------------------------------------
            # 3) Rank-based quantiles (stable)
            # ------------------------------------------------------------
            # Compute row_number over sorted energy + total count
            ranked = (
                session.query(
                    StepORM.total_energy.label("e"),
                    func.row_number().over(order_by=StepORM.total_energy.asc()).label("rn"),
                    func.count().over().label("n"),
                )
                .filter(StepORM.id.in_(select(step_q.subquery().c.id)))
                .subquery()
            )

            # target ranks: ceil(q * n)
            # SQLAlchemy doesn't have CEIL portable across sqlite; use CAST(q*n + 0.999999)
            def rank_expr(q):
                return func.cast(q * ranked.c.n + 0.999999, Integer)

            k_soft = rank_expr(q_soft)
            k_med = rank_expr(q_med)
            k_hard = rank_expr(q_hard)

            # select energy at those ranks
            tau_soft = session.query(ranked.c.e).filter(ranked.c.rn == k_soft).scalar()
            tau_med = session.query(ranked.c.e).filter(ranked.c.rn == k_med).scalar()
            tau_hard = session.query(ranked.c.e).filter(ranked.c.rn == k_hard).scalar()

            # Fallback if any scalar came back None (shouldn't happen, but be defensive)
            if tau_soft is None or tau_med is None or tau_hard is None:
                # fallback to min/median/max
                stats = (
                    session.query(func.min(StepORM.total_energy), func.avg(StepORM.total_energy), func.max(StepORM.total_energy))
                    .filter(StepORM.id.in_(select(step_q.subquery().c.id)))
                    .one()
                )
                mn, mean, mx = stats
                tau_soft = tau_soft if tau_soft is not None else mean
                tau_med = tau_med if tau_med is not None else mean
                tau_hard = tau_hard if tau_hard is not None else mx

            return ThresholdResult(
                thresholds=Thresholds(
                    float(tau_soft),
                    float(tau_med),
                    float(tau_hard),
                ),
                sample_count=int(n),
            )

        return self._run(op)