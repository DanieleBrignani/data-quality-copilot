"""Database engine and session management."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from functools import lru_cache

from sqlalchemy import Engine, create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker

from dqcopilot.config import Settings, get_settings
from dqcopilot.db.models import Base
from dqcopilot.logging_conf import get_logger

logger = get_logger(__name__)


class DatabaseUnavailableError(Exception):
    """The database could not be reached.

    The application treats persistence as optional: if the database is down the
    analysis still runs, only the audit trail is not written. This exception exists so
    the UI can say that clearly instead of showing a stack trace.
    """


def build_engine(settings: Settings | None = None) -> Engine:
    """Create a SQLAlchemy engine from the configured ``DATABASE_URL``."""
    settings = settings or get_settings()
    url = settings.database_url

    kwargs: dict[str, object] = {"future": True, "pool_pre_ping": True}
    if url.startswith("sqlite"):
        # SQLite in Streamlit is touched from several threads; the default check is
        # too strict for that, and the pool must not be shared across in-memory files.
        kwargs["connect_args"] = {"check_same_thread": False}
        kwargs.pop("pool_pre_ping")

    return create_engine(url, **kwargs)


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    """Return the process-wide engine."""
    return build_engine()


@lru_cache(maxsize=1)
def get_session_factory() -> sessionmaker[Session]:
    """Return the process-wide session factory."""
    return sessionmaker(bind=get_engine(), expire_on_commit=False, future=True)


def create_all(engine: Engine | None = None) -> None:
    """Create every table.

    Alembic owns the schema in Docker and in any real deployment; this helper exists for
    tests and for a first local run against SQLite.
    """
    Base.metadata.create_all(engine or get_engine())


def ensure_schema(settings: Settings | None = None, engine: Engine | None = None) -> bool:
    """Create the tables on a local SQLite database if they are missing.

    This is a **development convenience only**, and it deliberately does nothing for
    PostgreSQL: in Docker and in any real deployment Alembic owns the schema, and
    silently creating tables behind a migration tool is how schema drift starts. On
    SQLite the alternative is a first run that fails with "no such table", which is a
    poor welcome for someone who just cloned the repository.

    Returns:
        True when the schema is present and usable.
    """
    settings = settings or get_settings()
    if not settings.database_url.startswith("sqlite"):
        return check_connection(engine)

    try:
        create_all(engine or get_engine())
    except SQLAlchemyError as exc:
        logger.warning("Could not create the SQLite schema", extra={"reason": short_reason(exc)})
        return False
    return True


@contextmanager
def session_scope(factory: sessionmaker[Session] | None = None) -> Iterator[Session]:
    """Provide a transactional session that commits on success and rolls back on error.

    Raises:
        DatabaseUnavailableError: If the database cannot be reached or the transaction
            fails. The original error is chained for the logs.
    """
    session = (factory or get_session_factory())()
    try:
        yield session
        session.commit()
    except SQLAlchemyError as exc:
        session.rollback()
        logger.exception("Database transaction failed", extra={"error_type": type(exc).__name__})
        raise DatabaseUnavailableError(short_reason(exc)) from exc
    finally:
        session.close()


MAX_REASON_LENGTH = 160


def short_reason(exc: Exception) -> str:
    """Summarise a database error in one line, safe to show in the UI.

    A raw SQLAlchemy message embeds the failing statement and its bound parameters.
    Those belong in the log, not on screen: they are long, they leak schema detail, and
    the bound values can be derived from the user's data. The full exception is logged
    with a traceback by the caller.
    """
    original = getattr(exc, "orig", None)
    detail = str(original) if original is not None else str(exc)
    first_line = detail.strip().splitlines()[0] if detail.strip() else ""

    if len(first_line) > MAX_REASON_LENGTH:
        first_line = first_line[: MAX_REASON_LENGTH - 1].rstrip() + "…"
    return f"{type(exc).__name__}: {first_line}" if first_line else type(exc).__name__


def check_connection(engine: Engine | None = None) -> bool:
    """Return True when the database answers a trivial query."""
    try:
        with (engine or get_engine()).connect() as connection:
            connection.execute(text("SELECT 1"))
    except SQLAlchemyError as exc:
        logger.warning("Database unreachable", extra={"error_type": type(exc).__name__})
        return False
    return True


def reset_caches() -> None:
    """Drop the cached engine and session factory (used by the tests)."""
    get_engine.cache_clear()
    get_session_factory.cache_clear()
