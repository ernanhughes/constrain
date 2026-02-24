# constrain/data/stores/causal_evaluation_store.py

import time

from constrain.data.orm.causal_evaluation import CausalEvaluationORM
from constrain.data.schemas.causal_evaluation import CausalEvaluationDTO
from constrain.data.stores.base_store import BaseSQLAlchemyStore


class CausalEvaluationStore(BaseSQLAlchemyStore[CausalEvaluationDTO]):

    orm_model = CausalEvaluationORM
    

    def create(
        self,
        *,
        run_id,
        method,
        ate,
        ci_lower,
        ci_upper,
        n_samples,
    ):

        now = time.time()

        def op(s):
            obj = CausalEvaluationORM(
                run_id=run_id,
                method=method,
                ate=ate,
                ci_lower=ci_lower,
                ci_upper=ci_upper,
                n_samples=n_samples,
                created_at=now,
            )
            s.add(obj)
            s.flush()
            return obj

        row = self._run(op)
        return CausalEvaluationDTO.model_validate(row)