# verity_core/stores/base_store.py
from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from time import perf_counter
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    DefaultDict,
    Dict,
    Generic,
    List,
    Literal,
    Optional,
    Type,
    TypeVar,
)

from sqlalchemy import func
from sqlalchemy.orm import sessionmaker

from .db_scope import retry, session_scope

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

TDTO = TypeVar("TDTO")
HookPoint = Literal["after_create", "after_update", "after_delete"]
HookFn = Callable[["HookContext[TDTO]"], None]


@dataclass(frozen=True)
class HookContext(Generic[TDTO]):
    """Structured context for store hooks - prevents hidden dependencies"""

    memory: Any  # Memory instance for store access
    session: Session  # Current DB session (for same-transaction operations)
    store_name: str  # Name of triggering store (for logging)
    hook_point: HookPoint
    dto: TDTO  # The persisted DTO (with ID after flush)


class BaseSQLAlchemyStore(Generic[TDTO]):
    """
    Generic SQLAlchemy store with standard CRUD-ish helpers.

    IMPORTANT:
    - This base uses SHORT-LIVED sessions per call.
    - Pass a sessionmaker only (recommended).
    """

    orm_model: Type[Any] = None
    id_attr: str = "id"
    default_order_by: Optional[Any] = None

    def __init__(self, sm: sessionmaker, memory: Optional[Any] = None):
        assert self.orm_model is not None, "Subclasses must set orm_model"
        self.session: sessionmaker = sm
        self.memory = memory
        self.name = self.__class__.__name__
        self._hooks: DefaultDict[HookPoint, list[HookFn[Any]]] = defaultdict(list)
        self._hook_warn_ms = 25.0

    def _run(self, fn: Callable[[Any], Any], tries: int = 2):
        def op():
            with session_scope(self.session) as s:
                return fn(s)

        return retry(op, tries=tries)

    # Hook registration and execution
    def register_hook(self, hook_point: HookPoint, fn: HookFn[TDTO]) -> None:
        """Register hook function for specified lifecycle point"""
        self._hooks[hook_point].append(fn)

    def hook_count(self, hook_point: HookPoint) -> int:
        """Public API for health checks to verify hook registration."""
        return len(self._hooks.get(hook_point, []))

    def _run_hooks(self, hook_point: HookPoint, session: Session, dto: TDTO) -> None:
        """Execute all hooks for lifecycle point with failure isolation"""
        if not self.memory or not self._hooks.get(hook_point):
            return

        ctx = HookContext(
            memory=self.memory,
            session=session,
            store_name=self.name,
            hook_point=hook_point,
            dto=dto,
        )

        for fn in self._hooks[hook_point]:
            t0 = perf_counter()
            try:
                # 🔑 CRITICAL: Savepoint isolates hook failures
                with session.begin_nested():  # ← ROLLS BACK ONLY HOOK WORK ON FAILURE
                    fn(ctx)
            except Exception as e:
                # HARD RULE: never break persistence because a hook failed
                fn_name = getattr(fn, "__name__", repr(fn))
                logger.warning(
                    f"Store hook failed: store={self.name} hook={hook_point} fn={fn_name} err={e}",
                    exc_info=True,
                )
            finally:
                dt_ms = (perf_counter() - t0) * 1000.0
                if dt_ms > self._hook_warn_ms:
                    fn_name = getattr(fn, "__name__", repr(fn))
                    logger.debug(
                        f"Slow hook: store={self.name} hook={hook_point} fn={fn_name} took={dt_ms:.2f}ms"
                    )

    # -------- Standard APIs --------

    def get_by_id(self, obj_id: Any) -> Any | None:
        return self._run(lambda s: s.get(self.orm_model, obj_id))

    def count(self, **filters) -> int:
        return self._run(
            lambda s: int(
                (
                    s.query(func.count("*"))
                    .select_from(self.orm_model)
                    .filter_by(**filters)
                    if filters
                    else s.query(func.count("*")).select_from(self.orm_model)
                ).scalar()
                or 0
            )
        )

    def exists(self, **filters) -> bool:
        def op(s):
            q = s.query(self.orm_model)
            if filters:
                q = q.filter_by(**filters)
            return s.query(q.exists()).scalar() or False

        return self._run(op)

    def list(
        self,
        *,
        limit: int = 100,
        offset: int = 0,
        order_by: Optional[Any] = None,
        desc: bool = True,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Any]:
        def op(s):
            q = s.query(self.orm_model)
            if filters:
                q = q.filter_by(**filters)

            col = order_by or self.default_order_by
            if isinstance(col, str):
                col = getattr(self.orm_model, col, None)
            if col is not None:
                q = q.order_by(col.desc() if desc else col.asc())

            if offset:
                q = q.offset(offset)
            if limit:
                q = q.limit(limit)
            return q.all()

        return self._run(op)

    def delete_by_id(self, obj_id: Any) -> bool:
        def op(s):
            obj = s.get(self.orm_model, obj_id)
            if not obj:
                return False
            s.delete(obj)
            return True

        return self._run(op)

    def deactivate_by_id(self, obj_id: Any) -> bool:
        def op(s):
            obj = s.get(self.orm_model, obj_id)
            if not obj:
                return False
            if not hasattr(obj, "is_active"):
                raise AttributeError(
                    f"{self.orm_model.__name__} has no 'is_active' field"
                )
            obj.is_active = False
            return True

        return self._run(op)

    def bulk_add(self, items: List[Dict[str, Any]]) -> List[Any]:
        def op(s):
            objs = [self.orm_model(**item) for item in items]
            s.add_all(objs)
            s.flush()
            return objs

        return self._run(op)

    def save(self, obj: Any) -> Any:
        def op(s):
            s.add(obj)
            s.flush()
            return obj

        return self._run(op)

    def save_all(self, objs: List[Any]) -> List[Any]:
        def op(s):
            s.add_all(objs)
            s.flush()
            return objs

        return self._run(op)

    def get_one(
        self,
        *,
        order_by: Optional[Any] = None,
        desc: bool = True,
        filters: Optional[Dict[str, Any]] = None,
    ) -> Any | None:
        def op(s):
            q = s.query(self.orm_model)
            if filters:
                q = q.filter_by(**filters)

            col = order_by or self.default_order_by
            if isinstance(col, str):
                col = getattr(self.orm_model, col, None)
            if col is not None:
                q = q.order_by(col.desc() if desc else col.asc())

            return q.first()

        return self._run(op)

    def delete_where(self, **filters) -> int:
        def op(s):
            q = s.query(self.orm_model).filter_by(**filters)
            n = q.count()
            q.delete(synchronize_session=False)
            return int(n)

        return self._run(op)

    def _resolve_column(self, col: Any):
        """Allow passing either a column or a string column name."""
        if isinstance(col, str):
            resolved = getattr(self.orm_model, col, None)
            if resolved is None:
                raise AttributeError(f"{self.orm_model.__name__} has no column '{col}'")
            return resolved
        return col

    def count_since(
        self,
        since: datetime,
        *,
        time_col: Any = "created_at",
        filters: Optional[Dict[str, Any]] = None,
    ) -> int:
        """
        Count rows where time_col >= since, plus optional equality filters.

        Example:
            store.count_since(dt, time_col="created_at", filters={"goal_stream_id": 3})
        """
        time_col = self._resolve_column(time_col)

        def op(s):
            q = (
                s.query(func.count("*"))
                .select_from(self.orm_model)
                .filter(time_col >= since)
            )
            if filters:
                q = q.filter_by(**filters)
            return int(q.scalar() or 0)

        return self._run(op)

    def list_since(
        self,
        since: datetime,
        *,
        time_col: Any = "created_at",
        limit: int = 100,
        offset: int = 0,
        order_by: Optional[Any] = None,
        desc: bool = True,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Any]:
        """
        List rows where time_col >= since, plus optional equality filters.
        """
        time_col = self._resolve_column(time_col)

        def op(s):
            q = s.query(self.orm_model).filter(time_col >= since)
            if filters:
                q = q.filter_by(**filters)

            col = order_by or self.default_order_by or time_col
            col = self._resolve_column(col) if isinstance(col, str) else col
            if col is not None:
                q = q.order_by(col.desc() if desc else col.asc())

            if offset:
                q = q.offset(offset)
            if limit:
                q = q.limit(limit)
            return q.all()

        return self._run(op)


    def update(self, obj_id: Any, updates: Dict[str, Any]) -> Optional[TDTO]:
        """
        Update an object by ID with a dictionary of changes.
        Returns the updated DTO or None if not found.
        """

        def op(s: Session):
            # 1. Fetch the ORM object
            obj = s.get(self.orm_model, obj_id)
            if not obj:
                return None

            # 2. Apply updates
            for key, value in updates.items():
                if hasattr(obj, key):
                    setattr(obj, key, value)
                else:
                    logger.warning(
                        f"Store {self.name}: Attribute '{key}' not found on {self.orm_model.__name__}"
                    )

            s.flush()

            # 3. Convert to DTO and run hooks if they exist
            # This handles the mapping back to your Pydantic schemas
            if hasattr(self, "_to_dto"):
                dto = self._to_dto(obj)
                self._run_hooks("after_update", s, dto)
                return dto

            return obj

        return self._run(op)
