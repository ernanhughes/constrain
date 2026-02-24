from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import numpy as np
from sqlalchemy.orm import sessionmaker
from sqlalchemy import select

from constrain.data.orm.recovery_experiment import RecoveryExperimentORM
from constrain.data.schemas.recovery_experiment import RecoveryExperimentDTO
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


class RecoveryExperimentStore(BaseSQLAlchemyStore[RecoveryExperimentDTO]):
    orm_model = RecoveryExperimentORM
    default_order_by = "start_time"

    def __init__(self, sm: sessionmaker, memory: Optional[Any] = None):
        super().__init__(sm, memory)
        self.name = "recovery_experiments"

    @staticmethod
    def _to_dto(row: RecoveryExperimentORM) -> RecoveryExperimentDTO:
        """Convert ORM row to DTO (Pydantic handles JSON parsing via validators)."""
        return RecoveryExperimentDTO.model_validate(row)

    @staticmethod
    def _from_dto(dto: RecoveryExperimentDTO) -> RecoveryExperimentORM:
        """Convert DTO to ORM (Pydantic serializers handle JSON encoding)."""
        # Use model_dump to trigger field_serializers
        data = dto.model_dump()
        return RecoveryExperimentORM(
            experiment_name=data["experiment_name"],
            experiment_type=data["experiment_type"],
            run_ids=data["run_ids"],  # Already serialized by field_serializer
            problem_filter=data["problem_filter"],
            energy_threshold=data["energy_threshold"],
            accuracy_delta_threshold=data["accuracy_delta_threshold"],
            min_pre_intervention_steps=data["min_pre_intervention_steps"],
            min_post_intervention_steps=data["min_post_intervention_steps"],
            start_time=data["start_time"],
            end_time=data["end_time"],
            status=data["status"],
            summary_metrics=data["summary_metrics"],
            per_problem_results=data["per_problem_results"],
            statistical_tests=data["statistical_tests"],
            overlap_warning=data["overlap_warning"],
            confounding_warning=data["confounding_warning"],
            notes=data["notes"],
        )

    def create(self, dto: RecoveryExperimentDTO) -> RecoveryExperimentDTO:
        def op(s):
            obj = self._from_dto(dto)
            s.add(obj)
            s.flush()
            return self._to_dto(obj)
        return self._run(op)

    def update(self, experiment_id: int, updates: Dict[str, Any]) -> Optional[RecoveryExperimentDTO]:
        def op(s):
            obj = s.query(RecoveryExperimentORM).filter_by(id=experiment_id).first()
            if obj is None:
                return None

            for key, value in updates.items():
                if not hasattr(obj, key):
                    continue

                # Handle JSON fields: serialize with numpy support
                if key in ("run_ids", "problem_filter", "summary_metrics", "per_problem_results", "statistical_tests") and value is not None:
                    value = json.dumps(_json_serialize(value))

                setattr(obj, key, value)

            s.flush()
            return self._to_dto(obj)

        return self._run(op)

    def get_by_id(self, experiment_id: int) -> Optional[RecoveryExperimentDTO]:
        def op(s):
            stmt = select(self.orm_model).where(self.orm_model.id == experiment_id)
            row = s.execute(stmt).scalars().first()
            return self._to_dto(row) if row else None
        return self._run(op)

    def get_all(self, limit: Optional[int] = None) -> List[RecoveryExperimentDTO]:
        def op(s):
            stmt = select(self.orm_model).order_by(self.orm_model.start_time.desc())
            if limit:
                stmt = stmt.limit(limit)
            rows = s.execute(stmt).scalars().all()
            return [self._to_dto(row) for row in rows]
        return self._run(op)

    def get_by_status(self, status: str) -> List[RecoveryExperimentDTO]:
        def op(s):
            stmt = select(self.orm_model).where(self.orm_model.status == status)
            rows = s.execute(stmt).scalars().all()
            return [self._to_dto(row) for row in rows]
        return self._run(op)

    def get_by_run_id(self, run_id: str) -> List[RecoveryExperimentDTO]:
        """Get all recovery experiments that include a specific run."""
        def op(s):
            # Query experiments where run_ids contains the target run_id
            stmt = select(self.orm_model)
            rows = s.execute(stmt).scalars().all()
            
            # Filter in Python since run_ids is stored as JSON
            matching = []
            for row in rows:
                try:
                    run_ids = json.loads(row.run_ids) if isinstance(row.run_ids, str) else row.run_ids
                    if run_id in run_ids:
                        matching.append(self._to_dto(row))
                except (json.JSONDecodeError, TypeError):
                    continue
            return matching
        return self._run(op)

    def get_running(self) -> List[RecoveryExperimentDTO]:
        return self.get_by_status("running")

    def get_completed(self, limit: Optional[int] = None) -> List[RecoveryExperimentDTO]:
        def op(s):
            stmt = select(self.orm_model).where(
                self.orm_model.status == "completed"
            ).order_by(self.orm_model.start_time.desc())
            if limit:
                stmt = stmt.limit(limit)
            rows = s.execute(stmt).scalars().all()
            return [self._to_dto(row) for row in rows]
        return self._run(op)

    def complete(
        self,
        experiment_id: int,
        summary_metrics: Dict[str, Any],
        statistical_tests: Optional[Dict[str, Any]] = None,
        per_problem_results: Optional[Dict[str, Any]] = None,
        overlap_warning: bool = False,
        confounding_warning: bool = False,
    ) -> Optional[RecoveryExperimentDTO]:
        """Mark experiment as completed with results."""
        import time
        return self.update(experiment_id, {
            "status": "completed",
            "end_time": time.time(),
            "summary_metrics": summary_metrics,
            "statistical_tests": statistical_tests,
            "per_problem_results": per_problem_results,
            "overlap_warning": overlap_warning,
            "confounding_warning": confounding_warning,
        })

    def delete(self, experiment_id: int) -> bool:
        def op(s):
            obj = s.query(RecoveryExperimentORM).filter_by(id=experiment_id).first()
            if obj is None:
                return False
            s.delete(obj)
            s.flush()
            return True
        return self._run(op)