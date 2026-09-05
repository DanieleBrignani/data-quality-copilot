"""Pydantic models for proposed corrections and the human approval decision."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field

from dqcopilot.models.enums import CorrectionAction, DecisionStatus, FindingSource

MAX_PREVIEW_ROWS = 10


def make_proposal_id(finding_id: str, action: CorrectionAction, discriminator: str = "") -> str:
    """Build a stable identifier for a correction proposal."""
    raw = f"{finding_id}|{action.value}|{discriminator}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


class ChangePreview(BaseModel):
    """A single before/after example of what a correction would do."""

    row_index: int
    column: str
    before: str
    after: str


class CorrectionProposal(BaseModel):
    """A correction the system offers to apply, pending explicit human approval.

    Nothing in this application mutates a dataset until a proposal has been approved.
    """

    proposal_id: str
    finding_id: str
    action: CorrectionAction
    source: FindingSource = FindingSource.DETERMINISTIC

    column: str | None = None
    title: str
    description: str
    rationale: str = ""

    parameters: dict[str, Any] = Field(default_factory=dict)
    affected_rows: int = 0
    preview: list[ChangePreview] = Field(default_factory=list, max_length=MAX_PREVIEW_ROWS)

    destructive: bool = Field(
        default=False,
        description="True when the correction discards information (row drop, value clearing).",
    )
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)

    @property
    def requires_extra_care(self) -> bool:
        """True when the UI should visually warn before approval."""
        return self.destructive or self.source is FindingSource.AI


class CorrectionDecision(BaseModel):
    """The user's decision about one proposal."""

    proposal_id: str
    status: DecisionStatus = DecisionStatus.PENDING
    decided_at: datetime | None = None
    decided_by: str = "demo-user"
    note: str = ""

    @classmethod
    def approve(cls, proposal_id: str, decided_by: str = "demo-user") -> CorrectionDecision:
        """Return an approved decision stamped with the current UTC time."""
        return cls(
            proposal_id=proposal_id,
            status=DecisionStatus.APPROVED,
            decided_at=datetime.now(UTC),
            decided_by=decided_by,
        )

    @classmethod
    def reject(cls, proposal_id: str, decided_by: str = "demo-user") -> CorrectionDecision:
        """Return a rejected decision stamped with the current UTC time."""
        return cls(
            proposal_id=proposal_id,
            status=DecisionStatus.REJECTED,
            decided_at=datetime.now(UTC),
            decided_by=decided_by,
        )


class AppliedChange(BaseModel):
    """Record of one concrete change applied to the dataset, for the audit log."""

    proposal_id: str
    action: CorrectionAction
    column: str | None
    rows_changed: int
    parameters: dict[str, Any] = Field(default_factory=dict)
    applied_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
