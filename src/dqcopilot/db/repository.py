"""Persistence for analyses, decisions and the audit trail.

Every write goes through this module, which is also where the privacy rule is enforced:
:func:`redact_details` strips example values out of a finding before it reaches the
database. The in-memory finding keeps its examples so the UI and the downloadable report
can show them; only the persisted copy is reduced to counts and row numbers.
"""

from __future__ import annotations

import hashlib
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from dqcopilot import __version__
from dqcopilot.config import Settings, get_settings
from dqcopilot.db.models import (
    AiCallRecord,
    Analysis,
    AnalysisColumn,
    AppliedChangeRecord,
    AuditEvent,
    FindingRecord,
    ProposalRecord,
)
from dqcopilot.logging_conf import get_logger
from dqcopilot.models.enums import DecisionStatus
from dqcopilot.services.analysis import AnalysisResult
from dqcopilot.services.review import CleanedDataset, ReviewSession

logger = get_logger(__name__)

#: Keys inside ``Finding.details`` that may contain values copied out of the dataset.
SENSITIVE_DETAIL_KEYS: frozenset[str] = frozenset(
    {
        "examples",
        "variant_groups",
        "constant_value",
        "mapping",
        "values",
    }
)


def content_hash(content: bytes) -> str:
    """Return the SHA-256 of the uploaded bytes.

    Storing the hash rather than the file lets a repeated upload be recognised without
    the demo ever keeping someone's data.
    """
    return hashlib.sha256(content).hexdigest()


def redact_details(details: dict[str, Any], persist_examples: bool = False) -> dict[str, Any]:
    """Return a copy of ``details`` that is safe to store.

    Keys that can hold values copied out of the dataset are replaced by a count, so the
    audit trail records how many examples there were without recording what they said.
    """
    if persist_examples:
        return dict(details)

    redacted: dict[str, Any] = {}
    for key, value in details.items():
        if key not in SENSITIVE_DETAIL_KEYS:
            redacted[key] = value
        elif isinstance(value, (list, dict)):
            redacted[f"{key}_count"] = len(value)
        else:
            redacted[f"{key}_present"] = True
    return redacted


# --------------------------------------------------------------------------- writes


def save_analysis(
    session: Session,
    result: AnalysisResult,
    content: bytes | None = None,
    settings: Settings | None = None,
) -> Analysis:
    """Persist an analysis, its column profiles and its findings.

    Args:
        session: An open SQLAlchemy session.
        result: The analysis to store.
        content: The uploaded bytes, used only to compute a hash.
        settings: Optional settings override.

    Returns:
        The persisted :class:`~dqcopilot.db.models.Analysis` row.
    """
    settings = settings or get_settings()
    persist_examples = settings.persist_examples

    record = Analysis(
        analysis_id=result.analysis_id,
        created_at=result.created_at,
        source_name=result.source_name,
        content_hash=content_hash(content) if content is not None else None,
        row_count=result.row_count,
        column_count=result.column_count,
        size_bytes=result.size_bytes,
        quality_score=result.score.overall,
        score_breakdown={
            "overall": result.score.overall,
            "dataset_penalty": result.score.dataset_penalty,
            "severity_counts": result.score.severity_counts,
            "formula": result.score.formula,
            "columns": [
                {"column": entry.column, "score": entry.score, "penalty": entry.penalty}
                for entry in result.score.columns
            ],
        },
        duration_seconds=round(result.duration_seconds, 4),
        app_version=__version__,
        rule_set_name=result.rules.name if result.rules else None,
        notes=list(result.notes),
    )

    for column in result.profile.columns:
        record.columns.append(
            AnalysisColumn(
                name=column.name,
                position=column.position,
                pandas_dtype=column.pandas_dtype,
                semantic_type=column.semantic_type.value,
                semantic_type_confidence=column.semantic_type_confidence,
                row_count=column.row_count,
                missing_count=column.missing_count,
                unique_count=column.unique_count,
                column_score=result.score.column_score(column.name),
            )
        )

    for finding in result.findings.findings:
        record.findings.append(
            FindingRecord(
                finding_id=finding.finding_id,
                check_id=finding.check_id,
                issue_type=finding.issue_type.value,
                source=finding.source.value,
                severity=finding.severity.value,
                column_name=finding.column,
                title=finding.title[:500],
                explanation=finding.explanation,
                affected_rows=finding.affected_rows,
                row_count=finding.row_count,
                confidence=finding.confidence,
                details=redact_details(finding.details, persist_examples),
            )
        )

    session.add(record)
    session.flush()

    record.audit_events.append(
        AuditEvent(
            event_type="analysis.completed",
            summary=(
                f"Analysed '{result.source_name}': {result.row_count:,} rows, "
                f"{len(result.findings)} finding(s), score {result.score.overall:.1f}."
            ),
            payload={
                "rows": result.row_count,
                "columns": result.column_count,
                "findings": len(result.findings),
                "quality_score": result.score.overall,
            },
        )
    )
    session.flush()

    logger.info(
        "Analysis persisted",
        extra={
            "analysis_id": result.analysis_id,
            "findings": len(result.findings),
            "persist_examples": persist_examples,
        },
    )
    return record


def save_review(
    session: Session,
    review: ReviewSession,
    settings: Settings | None = None,
) -> Analysis:
    """Persist the proposals offered and the decision taken on each one.

    Rejections are stored as deliberately as approvals: an audit trail that only records
    what changed cannot answer whether a problem was seen and consciously left alone.
    """
    settings = settings or get_settings()
    record = require_analysis(session, review.analysis.analysis_id)

    existing = {proposal.proposal_id for proposal in record.proposals}
    for proposal in review.proposals:
        if proposal.proposal_id in existing:
            continue
        decision = review.decisions.get(proposal.proposal_id)
        record.proposals.append(
            ProposalRecord(
                proposal_id=proposal.proposal_id,
                finding_id=proposal.finding_id,
                action=proposal.action.value,
                source=proposal.source.value,
                column_name=proposal.column,
                title=proposal.title[:500],
                description=proposal.description,
                parameters=redact_details(proposal.parameters, settings.persist_examples),
                affected_rows=proposal.affected_rows,
                destructive=proposal.destructive,
                confidence=proposal.confidence,
                decision_status=(decision.status if decision else DecisionStatus.PENDING).value,
                decided_at=decision.decided_at if decision else None,
                decided_by=decision.decided_by if decision else None,
                decision_note=decision.note if decision else "",
            )
        )

    counts = review.decision_counts()
    record.audit_events.append(
        AuditEvent(
            event_type="review.decided",
            actor=review.reviewer,
            summary=(
                f"{counts['approved']} correction(s) approved, {counts['rejected']} rejected, "
                f"{counts['pending']} left pending."
            ),
            payload=counts,
        )
    )
    session.flush()
    return record


def save_cleaned_dataset(
    session: Session,
    review: ReviewSession,
    cleaned: CleanedDataset,
    settings: Settings | None = None,
) -> Analysis:
    """Persist the changes that were actually applied to the dataset."""
    settings = settings or get_settings()
    record = require_analysis(session, review.analysis.analysis_id)

    for change in cleaned.applied:
        record.applied_changes.append(
            AppliedChangeRecord(
                proposal_id=change.proposal_id,
                action=change.action.value,
                column_name=change.column,
                rows_changed=change.rows_changed,
                parameters=redact_details(change.parameters, settings.persist_examples),
                applied_at=change.applied_at,
            )
        )

    record.audit_events.append(
        AuditEvent(
            event_type="dataset.cleaned",
            actor=review.reviewer,
            summary=(
                f"Applied {len(cleaned.applied)} correction(s); "
                f"{cleaned.total_cells_changed:,} cell(s) changed, "
                f"{cleaned.rows_removed:,} row(s) removed; score "
                f"{cleaned.score_before.overall:.1f} to {cleaned.score_after.overall:.1f}."
            ),
            payload={
                "changes_applied": len(cleaned.applied),
                "cells_changed": cleaned.total_cells_changed,
                "rows_before": cleaned.rows_before,
                "rows_after": cleaned.rows_after,
                "score_before": cleaned.score_before.overall,
                "score_after": cleaned.score_after.overall,
                "score_delta": cleaned.score_delta,
            },
        )
    )
    session.flush()
    return record


def record_ai_call(
    session: Session,
    analysis_id: str | None,
    task: str,
    model: str,
    input_tokens: int,
    output_tokens: int,
    latency_ms: int,
    succeeded: bool = True,
    error_type: str | None = None,
) -> AiCallRecord:
    """Record the cost and latency of one Anthropic call, never its content."""
    record = AiCallRecord(
        task=task,
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        latency_ms=latency_ms,
        succeeded=succeeded,
        error_type=error_type,
    )
    if analysis_id:
        analysis = find_analysis(session, analysis_id)
        if analysis is not None:
            record.analysis_pk = analysis.id
    session.add(record)
    session.flush()
    return record


def log_event(
    session: Session,
    event_type: str,
    summary: str,
    analysis_id: str | None = None,
    actor: str = "demo-user",
    payload: dict[str, Any] | None = None,
) -> AuditEvent:
    """Append one event to the audit trail."""
    event = AuditEvent(
        event_type=event_type,
        summary=summary[:500],
        actor=actor,
        payload=payload or {},
    )
    if analysis_id:
        analysis = find_analysis(session, analysis_id)
        if analysis is not None:
            event.analysis_pk = analysis.id
    session.add(event)
    session.flush()
    return event


# --------------------------------------------------------------------------- reads


def find_analysis(session: Session, analysis_id: str) -> Analysis | None:
    """Return one analysis by its public id, or ``None``."""
    return session.scalar(select(Analysis).where(Analysis.analysis_id == analysis_id))


def require_analysis(session: Session, analysis_id: str) -> Analysis:
    """Return one analysis by its public id.

    Raises:
        LookupError: If no analysis with that id has been stored.
    """
    record = find_analysis(session, analysis_id)
    if record is None:
        raise LookupError(f"No stored analysis with id '{analysis_id}'.")
    return record


def recent_analyses(session: Session, limit: int = 20) -> list[Analysis]:
    """Return the most recent analyses, newest first."""
    return list(
        session.scalars(
            select(Analysis).order_by(Analysis.created_at.desc(), Analysis.id.desc()).limit(limit)
        )
    )


def audit_trail(session: Session, analysis_id: str) -> list[AuditEvent]:
    """Return every audit event for an analysis, oldest first."""
    record = require_analysis(session, analysis_id)
    return list(
        session.scalars(
            select(AuditEvent)
            .where(AuditEvent.analysis_pk == record.id)
            .order_by(AuditEvent.occurred_at.asc(), AuditEvent.id.asc())
        )
    )


def approved_changes(session: Session, analysis_id: str) -> list[AppliedChangeRecord]:
    """Return every change applied to a dataset."""
    record = require_analysis(session, analysis_id)
    return list(
        session.scalars(
            select(AppliedChangeRecord)
            .where(AppliedChangeRecord.analysis_pk == record.id)
            .order_by(AppliedChangeRecord.applied_at.asc(), AppliedChangeRecord.id.asc())
        )
    )


def ai_usage_totals(session: Session) -> dict[str, int]:
    """Return aggregate Anthropic token usage across every stored call."""
    row = session.execute(
        select(
            func.count(AiCallRecord.id),
            func.coalesce(func.sum(AiCallRecord.input_tokens), 0),
            func.coalesce(func.sum(AiCallRecord.output_tokens), 0),
        )
    ).one()
    return {"calls": int(row[0]), "input_tokens": int(row[1]), "output_tokens": int(row[2])}
