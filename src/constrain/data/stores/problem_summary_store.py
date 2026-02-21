from typing import Any, Optional

from sqlalchemy.orm import sessionmaker

from constrain.data.orm.problem_summary import ProblemSummaryORM
from constrain.data.stores.base_store import BaseSQLAlchemyStore


class ProblemSummaryStore(BaseSQLAlchemyStore[ProblemSummaryORM]):

    orm_model = ProblemSummaryORM
    default_order_by = "timestamp"

    def __init__(self, sm: sessionmaker, memory: Optional[Any] = None):
        super().__init__(sm, memory)
        self.name = "problem_summaries"


    def get_by_run(self, run_id: str):
        return self.list(filters={"run_id": run_id})