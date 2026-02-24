from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import numpy as np
from sqlalchemy.orm import sessionmaker
from sqlalchemy import select

from constrain.data.orm.experiment import ExperimentORM
from constrain.data.schemas.experiment import ExperimentDTO
from constrain.data.stores.base_store import BaseSQLAlchemyStore


def _json_serialize(obj: Any) -> Any:
    """
    Recursively convert numpy types to native Python for JSON serialization.
    """
    if isinstance(obj, dict):
        return {k: _json_serialize(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_json_serialize(item) for item in obj]
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    else:
        # Fallback: try string conversion
        return str(obj)


class ExperimentStore(BaseSQLAlchemyStore[ExperimentDTO]):
    orm_model = ExperimentORM
    default_order_by = "start_time"

    def __init__(self, sm: sessionmaker, memory: Optional[Any] = None):
        super().__init__(sm, memory)
        self.name = "experiments"

    @staticmethod
    def _to_dto(row: ExperimentORM) -> ExperimentDTO:
        """Convert ORM row to DTO (Pydantic handles JSON parsing via validators)."""
        return ExperimentDTO.model_validate(row)

    @staticmethod
    def _from_dto(dto: ExperimentDTO) -> ExperimentORM:
        """Convert DTO to ORM (Pydantic serializers handle JSON encoding)."""
        # Use model_dump to trigger field_serializers
        data = dto.model_dump()
        return ExperimentORM(
            experiment_name=data["experiment_name"],
            experiment_type=data["experiment_type"],
            policy_ids=data["policy_ids"],  # Already serialized by field_serializer
            seeds=data["seeds"],  # Already serialized by field_serializer
            num_problems=data["num_problems"],
            num_recursions=data["num_recursions"],
            initial_temperature=data["initial_temperature"],
            start_time=data["start_time"],
            end_time=data["end_time"],
            status=data["status"],
            results_summary=data["results_summary"],  # Already serialized by field_serializer
            notes=data["notes"],
            git_commit=data["git_commit"],
        )

    def create(self, dto: ExperimentDTO) -> ExperimentDTO:
        def op(s):
            obj = self._from_dto(dto)
            s.add(obj)
            s.flush()
            return self._to_dto(obj)
        return self._run(op)

    def update(self, experiment_id: int, updates: Dict[str, Any]) -> Optional[ExperimentDTO]:
        def op(s):
            obj = s.query(ExperimentORM).filter_by(id=experiment_id).first()
            if obj is None:
                return None

            for key, value in updates.items():
                if not hasattr(obj, key):
                    continue

                # Handle JSON fields: serialize with numpy support
                if key in ("policy_ids", "seeds", "results_summary") and value is not None:
                    value = json.dumps(_json_serialize(value))

                setattr(obj, key, value)

            s.flush()
            return self._to_dto(obj)

        return self._run(op)

    def get_by_id(self, experiment_id: int) -> Optional[ExperimentDTO]:
        def op(s):
            stmt = select(self.orm_model).where(self.orm_model.id == experiment_id)
            row = s.execute(stmt).scalars().first()
            return self._to_dto(row) if row else None
        return self._run(op)

    def get_all(self, limit: Optional[int] = None) -> List[ExperimentDTO]:
        def op(s):
            stmt = select(self.orm_model).order_by(self.orm_model.start_time.desc())
            if limit:
                stmt = stmt.limit(limit)
            rows = s.execute(stmt).scalars().all()
            return [self._to_dto(row) for row in rows]
        return self._run(op)

    def get_by_status(self, status: str) -> List[ExperimentDTO]:
        def op(s):
            stmt = select(self.orm_model).where(self.orm_model.status == status)
            rows = s.execute(stmt).scalars().all()
            return [self._to_dto(row) for row in rows]
        return self._run(op)

    def get_running(self) -> List[ExperimentDTO]:
        return self.get_by_status("running")

    def get_completed(self, limit: Optional[int] = None) -> List[ExperimentDTO]:
        def op(s):
            stmt = select(self.orm_model).where(
                self.orm_model.status == "completed"
            ).order_by(self.orm_model.start_time.desc())
            if limit:
                stmt = stmt.limit(limit)
            rows = s.execute(stmt).scalars().all()
            return [self._to_dto(row) for row in rows]
        return self._run(op)

    def complete(self, experiment_id: int, results_summary: Dict[str, Any]) -> Optional[ExperimentDTO]:
        """Mark experiment as completed with results."""
        import time
        return self.update(experiment_id, {
            "status": "completed",
            "end_time": time.time(),
            "results_summary": results_summary,
        })