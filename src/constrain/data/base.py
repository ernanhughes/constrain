from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

from constrain.config import get_config

engine = create_engine(
    get_config().db_url,
    future=True,
    connect_args={"check_same_thread": False},
)

SessionLocal = sessionmaker(
    bind=engine,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
    future=True,
)

Base = declarative_base()
