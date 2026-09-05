"""SQLAlchemy models for analyses, corrections and the audit log.

Privacy note
------------
These tables deliberately store **metadata about** the data, never the data itself.
Findings keep counts, ratios and row numbers; the example values a finding carries in
memory are stripped before persistence unless ``PERSIST_EXAMPLES`` is switched on. The
uploaded file is never written to disk or to the database. The consequence is that a
stored analysis can tell you *that* five email addresses were malformed and in which
rows, but not what they were - which is the right trade-off for a demo people are
invited to try.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

#: JSON on SQLite, JSONB on PostgreSQL. Keeping one type alias means the models work
#: unchanged against the SQLite used by the tests and the PostgreSQL used by Docker.
JSONType = JSON().with_variant(JSONB(), "postgresql")


def utcnow() -> datetime:
    """Return the current time as a timezone-aware UTC datetime."""
    return datetime.now(UTC)


class Base(DeclarativeBase):
    """Declarative base for every table."""


class Analysis(Base):
    """One analysis run over one uploaded file."""

    __tablename__ = "analyses"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    analysis_id: Mapped[str] = mapped_column(String(32), unique=True, index=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, index=True
    )

    source_name: Mapped[str] = mapped_column(String(255))
    #: SHA-256 of the uploaded bytes. Lets the same file be recognised across runs
    #: without keeping the file itself.
    content_hash: Mapped[str | None] = mapped_column(String(64), index=True, default=None)

    row_count: Mapped[int] = mapped_column(Integer, default=0)
    column_count: Mapped[int] = mapped_column(Integer, default=0)
    size_bytes: Mapped[int] = mapped_column(Integer, default=0)

    quality_score: Mapped[float] = mapped_column(Float, default=0.0)
    score_breakdown: Mapped[dict[str, Any]] = mapped_column(JSONType, default=dict)
    duration_seconds: Mapped[float] = mapped_column(Float, default=0.0)

    app_version: Mapped[str] = mapped_column(String(32), default="0.0.0")
    rule_set_name: Mapped[str | None] = mapped_column(String(120), default=None)
    notes: Mapped[list[str]] = mapped_column(JSONType, default=list)

    columns: Mapped[list[AnalysisColumn]] = relationship(
        back_populates="analysis", cascade="all, delete-orphan"
    )
    findings: Mapped[list[FindingRecord]] = relationship(
        back_populates="analysis", cascade="all, delete-orphan"
    )
    proposals: Mapped[list[ProposalRecord]] = relationship(
        back_populates="analysis", cascade="all, delete-orphan"
    )
    applied_changes: Mapped[list[AppliedChangeRecord]] = relationship(
        back_populates="analysis", cascade="all, delete-orphan"
    )
    audit_events: Mapped[list[AuditEvent]] = relationship(
        back_populates="analysis", cascade="all, delete-orphan"
    )


class AnalysisColumn(Base):
    """Per-column profile, kept so quality can be tracked over time."""

    __tablename__ = "analysis_columns"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    analysis_pk: Mapped[int] = mapped_column(
        ForeignKey("analyses.id", ondelete="CASCADE"), index=True
    )

    name: Mapped[str] = mapped_column(String(255))
    position: Mapped[int] = mapped_column(Integer, default=0)
    pandas_dtype: Mapped[str] = mapped_column(String(64), default="")
    semantic_type: Mapped[str] = mapped_column(String(32), default="unknown")
    semantic_type_confidence: Mapped[float] = mapped_column(Float, default=0.0)

    row_count: Mapped[int] = mapped_column(Integer, default=0)
    missing_count: Mapped[int] = mapped_column(Integer, default=0)
    unique_count: Mapped[int] = mapped_column(Integer, default=0)
    column_score: Mapped[float] = mapped_column(Float, default=100.0)

    analysis: Mapped[Analysis] = relationship(back_populates="columns")


class FindingRecord(Base):
    """A data quality finding, stripped of example values."""

    __tablename__ = "findings"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    analysis_pk: Mapped[int] = mapped_column(
        ForeignKey("analyses.id", ondelete="CASCADE"), index=True
    )

    finding_id: Mapped[str] = mapped_column(String(32), index=True)
    check_id: Mapped[str] = mapped_column(String(64), index=True)
    issue_type: Mapped[str] = mapped_column(String(64), index=True)
    source: Mapped[str] = mapped_column(String(16), default="deterministic", index=True)
    severity: Mapped[str] = mapped_column(String(16), default="medium", index=True)

    column_name: Mapped[str | None] = mapped_column(String(255), default=None)
    title: Mapped[str] = mapped_column(String(500), default="")
    explanation: Mapped[str] = mapped_column(Text, default="")

    affected_rows: Mapped[int] = mapped_column(Integer, default=0)
    row_count: Mapped[int] = mapped_column(Integer, default=0)
    confidence: Mapped[float] = mapped_column(Float, default=1.0)
    details: Mapped[dict[str, Any]] = mapped_column(JSONType, default=dict)

    analysis: Mapped[Analysis] = relationship(back_populates="findings")

    __table_args__ = (Index("ix_findings_analysis_severity", "analysis_pk", "severity"),)


class ProposalRecord(Base):
    """A correction that was offered, together with the decision taken on it."""

    __tablename__ = "correction_proposals"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    analysis_pk: Mapped[int] = mapped_column(
        ForeignKey("analyses.id", ondelete="CASCADE"), index=True
    )

    proposal_id: Mapped[str] = mapped_column(String(32), index=True)
    finding_id: Mapped[str] = mapped_column(String(32), index=True)
    action: Mapped[str] = mapped_column(String(48), index=True)
    source: Mapped[str] = mapped_column(String(16), default="deterministic")

    column_name: Mapped[str | None] = mapped_column(String(255), default=None)
    title: Mapped[str] = mapped_column(String(500), default="")
    description: Mapped[str] = mapped_column(Text, default="")
    parameters: Mapped[dict[str, Any]] = mapped_column(JSONType, default=dict)

    affected_rows: Mapped[int] = mapped_column(Integer, default=0)
    destructive: Mapped[bool] = mapped_column(Boolean, default=False)
    confidence: Mapped[float] = mapped_column(Float, default=1.0)

    # --- decision -----------------------------------------------------------
    decision_status: Mapped[str] = mapped_column(String(16), default="pending", index=True)
    decided_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), default=None)
    decided_by: Mapped[str | None] = mapped_column(String(120), default=None)
    decision_note: Mapped[str] = mapped_column(Text, default="")

    analysis: Mapped[Analysis] = relationship(back_populates="proposals")


class AppliedChangeRecord(Base):
    """A change that was actually written to the cleaned dataset."""

    __tablename__ = "applied_changes"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    analysis_pk: Mapped[int] = mapped_column(
        ForeignKey("analyses.id", ondelete="CASCADE"), index=True
    )

    proposal_id: Mapped[str] = mapped_column(String(32), index=True)
    action: Mapped[str] = mapped_column(String(48))
    column_name: Mapped[str | None] = mapped_column(String(255), default=None)
    rows_changed: Mapped[int] = mapped_column(Integer, default=0)
    parameters: Mapped[dict[str, Any]] = mapped_column(JSONType, default=dict)
    applied_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    analysis: Mapped[Analysis] = relationship(back_populates="applied_changes")


class AuditEvent(Base):
    """Append-only record of everything that happened to an analysis.

    Rows are only ever inserted. Nothing in the application updates or deletes an audit
    event, which is what makes the trail trustworthy.
    """

    __tablename__ = "audit_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    analysis_pk: Mapped[int | None] = mapped_column(
        ForeignKey("analyses.id", ondelete="CASCADE"), index=True, default=None
    )

    occurred_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, index=True
    )
    event_type: Mapped[str] = mapped_column(String(64), index=True)
    actor: Mapped[str] = mapped_column(String(120), default="demo-user")
    summary: Mapped[str] = mapped_column(String(500), default="")
    payload: Mapped[dict[str, Any]] = mapped_column(JSONType, default=dict)

    analysis: Mapped[Analysis | None] = relationship(back_populates="audit_events")


class AiCallRecord(Base):
    """Token usage and latency for one Anthropic API call.

    Prompts and responses are **not** stored: only the shape of the call, so cost and
    latency can be reviewed without keeping anything derived from the user's data.
    """

    __tablename__ = "ai_calls"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    analysis_pk: Mapped[int | None] = mapped_column(
        ForeignKey("analyses.id", ondelete="CASCADE"), index=True, default=None
    )

    occurred_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    task: Mapped[str] = mapped_column(String(64), index=True)
    model: Mapped[str] = mapped_column(String(64), default="")
    input_tokens: Mapped[int] = mapped_column(Integer, default=0)
    output_tokens: Mapped[int] = mapped_column(Integer, default=0)
    latency_ms: Mapped[int] = mapped_column(Integer, default=0)
    succeeded: Mapped[bool] = mapped_column(Boolean, default=True)
    error_type: Mapped[str | None] = mapped_column(String(120), default=None)
