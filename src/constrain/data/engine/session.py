# constrain_core/db/session.py
from __future__ import annotations

from pathlib import Path
from typing import Callable

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import NullPool

from constrain.data.base import Base


def default_sqlite_url(repo_root: str) -> str:
    """
    Creates/uses a local SQLite DB at: <repo_root>/.constrain/constrain.db
    """
    p = Path(repo_root) / ".constrain"
    p.mkdir(parents=True, exist_ok=True)
    db_path = p / "constrain.db"
    return f"sqlite:///{db_path.as_posix()}"


def _configure_sqlite_pragmas(engine: Engine, sqlite_busy_timeout_ms: int = 5000) -> None:
    """
    Apply SQLite pragmas on every DB-API connection.
    WAL + busy_timeout are the two biggest lock reducers on Windows.
    """

    @event.listens_for(engine, "connect")
    def _set_sqlite_pragmas(dbapi_connection, _connection_record) -> None:
        cur = dbapi_connection.cursor()
        try:
            # Safety + integrity
            cur.execute("PRAGMA foreign_keys=ON;")

            # Concurrency: WAL reduces writer blocking readers
            cur.execute("PRAGMA journal_mode=WAL;")

            # Wait a bit instead of throwing 'database is locked'
            cur.execute(f"PRAGMA busy_timeout={int(sqlite_busy_timeout_ms)};")  # ms

            # Performance vs durability tradeoff (reasonable default for dev tools)
            cur.execute("PRAGMA synchronous=NORMAL;")

            # Optional (safe) perf improvements:
            cur.execute("PRAGMA temp_store=MEMORY;")
        finally:
            cur.close()


def make_engine(
    db_url: str,
    *,
    echo: bool = False,
    sqlite_timeout_s: int = 30,
    sqlite_busy_timeout_ms: int = 5000,
    sqlite_use_null_pool: bool = True,
) -> Engine:
    """
    Create a SQLAlchemy engine with strong SQLite settings.

    Notes:
      - For SQLite, we default to NullPool to avoid lingering pooled connections
        that can increase locking issues in CLI / short-lived runs.
      - If you run a long-lived server process and want pooling, set
        sqlite_use_null_pool=False.
    """
    if db_url.startswith("sqlite:"):
        engine = create_engine(
            db_url,
            echo=echo,
            future=True,
            connect_args={
                "check_same_thread": False,
                "timeout": int(sqlite_timeout_s),  # seconds (sqlite busy handler)
            },
            poolclass=NullPool if sqlite_use_null_pool else None,
            pool_pre_ping=True,
        )

        _configure_sqlite_pragmas(engine, sqlite_busy_timeout_ms=sqlite_busy_timeout_ms)
        return engine

    # Postgres / others
    return create_engine(db_url, echo=echo, future=True, pool_pre_ping=True)


def make_session_factory(engine: Engine) -> sessionmaker:
    """
    Preferred factory: returns a sessionmaker.

    IMPORTANT: pass this sessionmaker into Stores, and let stores create short-lived
    sessions per operation (avoids long transactions -> fewer SQLite locks).
    """
    return sessionmaker(
        bind=engine,
        autoflush=False,
        autocommit=False,
        expire_on_commit=False,
        future=True,
    )


# Backwards-compat naming (older code imported SessionFactory / create_session_factory)
SessionFactory = make_session_factory


def create_session_factory(engine: Engine) -> Callable[[], Session]:
    """
    Backwards-compatible helper that returns a callable producing Sessions.
    """
    sm = make_session_factory(engine)
    return sm  # sessionmaker is callable -> returns Session



def create_session_factory_from_url(db_url: str, *, echo: bool = False) -> sessionmaker:
    engine = make_engine(db_url, echo=echo)

    # 🔥 ENSURE ORM MODELS ARE REGISTERED
    _import_orm_models()

    # 🔥 AUTO-CREATE TABLES
    Base.metadata.create_all(engine)

    return make_session_factory(engine)


def new_session(db_url: str, *, echo: bool = False) -> Session:
    """
    Convenience function for scripts/tests.

    For app code: build engine once, build sessionmaker once, pass sessionmaker to Stores.
    """
    engine = make_engine(db_url, echo=echo)
    sm = make_session_factory(engine)
    return sm()

def _import_orm_models():
    """
    Ensures all ORM models are imported so that
    Base.metadata knows about them before create_all().
    """
    from constrain.data.orm.calibration import CalibrationORM
    from constrain.data.orm.derived_metrics import DerivedMetricsORM
    from constrain.data.orm.embedding import EmbeddingORM
    from constrain.data.orm.experiment import ExperimentORM
    from constrain.data.orm.intervention import InterventionORM
    from constrain.data.orm.metric import MetricORM
    from constrain.data.orm.problem_summary import ProblemSummaryORM
    from constrain.data.orm.run import RunORM
    from constrain.data.orm.signal_discovery import SignalDiscoveryORM
    from constrain.data.orm.signal_report import SignalReportORM
    from constrain.data.orm.step import StepORM
