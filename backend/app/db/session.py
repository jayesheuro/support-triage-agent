"""
app/db/session.py

Responsibility: SQLAlchemy database engine, session factory, connection lifecycle,
and table initialization on startup.
"""

from __future__ import annotations

from collections.abc import Generator
from pathlib import Path

from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session, sessionmaker

from app.db.models import Base, Setting

_BACKEND_ROOT = Path(__file__).resolve().parent.parent.parent
_DATA_DIR = _BACKEND_ROOT / "data"
_DATA_DIR.mkdir(parents=True, exist_ok=True)
_DB_PATH = _DATA_DIR / "app.db"

DATABASE_URL = f"sqlite:///{_DB_PATH}"

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},
)


@event.listens_for(engine, "connect")
def _set_sqlite_pragma(dbapi_connection: object, connection_record: object) -> None:
    """Enforce WAL mode and foreign keys on SQLite connections."""
    cursor = dbapi_connection.cursor()  # type: ignore[attr-defined]
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()


SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency for yielding a scoped database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db(bind_engine=engine) -> None:
    """Create all tables and seed default settings if not already present."""
    Base.metadata.create_all(bind=bind_engine)

    session_cls = sessionmaker(autocommit=False, autoflush=False, bind=bind_engine)
    with session_cls() as session:
        auto_send_setting = session.query(Setting).filter_by(key="auto_send_enabled").first()
        if not auto_send_setting:
            session.add(
                Setting(
                    key="auto_send_enabled",
                    value="false",
                    description="Master switch for automated response delivery (flips via eval gate)",
                )
            )
            session.commit()
