"""Persistence as an optional capability.

Analysis is the product; the audit trail is a record of it. If PostgreSQL is down the
user should still be able to profile a file, review corrections and download a cleaned
dataset - they just will not get a stored history. Every function here therefore returns
a :class:`PersistenceOutcome` instead of raising, and the UI reports the degraded state
plainly rather than failing the whole page.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from sqlalchemy.orm import Session, sessionmaker

from dqcopilot.ai.suggester import AiSuggestions
from dqcopilot.config import Settings, get_settings
from dqcopilot.db import repository
from dqcopilot.db.session import DatabaseUnavailableError, get_session_factory, session_scope
from dqcopilot.logging_conf import get_logger
from dqcopilot.services.analysis import AnalysisResult
from dqcopilot.services.review import CleanedDataset, ReviewSession

logger = get_logger(__name__)


@dataclass(slots=True)
class PersistenceOutcome:
    """Whether a write succeeded, and why not when it did not."""

    stored: bool
    detail: str = ""

    @property
    def failed(self) -> bool:
        """True when the write did not happen."""
        return not self.stored


_SKIPPED = PersistenceOutcome(stored=False, detail="Persistence is disabled by configuration.")


def _run(
    operation: str,
    action: Callable[[Session], None],
    settings: Settings | None = None,
    factory: sessionmaker[Session] | None = None,
) -> PersistenceOutcome:
    """Run ``action`` inside a transaction, converting failures into an outcome."""
    settings = settings or get_settings()
    if not settings.persistence_enabled:
        return _SKIPPED

    try:
        with session_scope(factory or get_session_factory()) as session:
            action(session)
    except DatabaseUnavailableError as exc:
        logger.warning("Persistence skipped", extra={"operation": operation})
        return PersistenceOutcome(
            stored=False,
            detail=f"The database is unavailable, so {operation} was not recorded. ({exc})",
        )
    except LookupError as exc:
        logger.warning("Persistence target missing", extra={"operation": operation})
        return PersistenceOutcome(stored=False, detail=str(exc))

    return PersistenceOutcome(stored=True)


def store_analysis(
    result: AnalysisResult,
    content: bytes | None = None,
    settings: Settings | None = None,
    factory: sessionmaker[Session] | None = None,
) -> PersistenceOutcome:
    """Store an analysis and its findings."""
    settings = settings or get_settings()

    def action(session: Session) -> None:
        repository.save_analysis(session, result, content=content, settings=settings)

    return _run("the analysis", action, settings, factory)


def store_review(
    review: ReviewSession,
    settings: Settings | None = None,
    factory: sessionmaker[Session] | None = None,
) -> PersistenceOutcome:
    """Store the correction proposals and the decisions taken on them."""
    settings = settings or get_settings()

    def action(session: Session) -> None:
        repository.save_review(session, review, settings=settings)

    return _run("your review decisions", action, settings, factory)


def store_cleaned_dataset(
    review: ReviewSession,
    cleaned: CleanedDataset,
    settings: Settings | None = None,
    factory: sessionmaker[Session] | None = None,
) -> PersistenceOutcome:
    """Store the changes that were actually applied."""
    settings = settings or get_settings()

    def action(session: Session) -> None:
        repository.save_review(session, review, settings=settings)
        repository.save_cleaned_dataset(session, review, cleaned, settings=settings)

    return _run("the applied corrections", action, settings, factory)


def store_download(
    review: ReviewSession,
    artefact: str,
    settings: Settings | None = None,
    factory: sessionmaker[Session] | None = None,
) -> PersistenceOutcome:
    """Record that the user downloaded an artefact."""
    settings = settings or get_settings()

    def action(session: Session) -> None:
        repository.log_event(
            session,
            event_type="artefact.downloaded",
            summary=f"Downloaded the {artefact}.",
            analysis_id=review.analysis.analysis_id,
            actor=review.reviewer,
            payload={"artefact": artefact},
        )

    return _run("the download", action, settings, factory)


def store_ai_calls(
    review: ReviewSession,
    suggestions: AiSuggestions,
    settings: Settings | None = None,
    factory: sessionmaker[Session] | None = None,
) -> PersistenceOutcome:
    """Record the cost and latency of each Anthropic call made for this analysis.

    Only the shape of the call is stored - task, model, token counts, latency and
    whether it succeeded. Prompts and responses are never persisted, so the trail can
    answer "what did the AI cost and how slow was it" without keeping anything derived
    from the user's data.
    """
    settings = settings or get_settings()
    if not suggestions.calls:
        return PersistenceOutcome(stored=True, detail="No AI call to record.")

    def action(session: Session) -> None:
        for call in suggestions.calls:
            repository.record_ai_call(
                session,
                analysis_id=review.analysis.analysis_id,
                task=call.task,
                model=call.model,
                input_tokens=call.input_tokens,
                output_tokens=call.output_tokens,
                latency_ms=call.latency_ms,
                succeeded=call.succeeded,
                error_type=call.error_type,
            )
        repository.log_event(
            session,
            event_type="ai.suggestions_generated",
            summary=(
                f"{len(suggestions.calls)} AI call(s): "
                f"{suggestions.usage.get('input_tokens', 0):,} input and "
                f"{suggestions.usage.get('output_tokens', 0):,} output tokens, "
                f"{len(suggestions.findings)} suggestion(s), "
                f"{len(suggestions.rejected)} discarded by grounding."
            ),
            analysis_id=review.analysis.analysis_id,
            actor=review.reviewer,
            payload=dict(suggestions.usage),
        )

    return _run("the AI usage", action, settings, factory)
