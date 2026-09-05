"""Pydantic models for data quality findings."""

from __future__ import annotations

import hashlib
from typing import Any

from pydantic import BaseModel, Field

from dqcopilot.models.enums import FindingSource, IssueType, Severity

MAX_ROW_SAMPLE = 200


def make_finding_id(
    source: FindingSource,
    check_id: str,
    issue_type: IssueType,
    column: str | None,
    fingerprint: str = "",
) -> str:
    """Build a stable, reproducible identifier for a finding.

    The identifier must be deterministic so that a user's approval decisions can be
    matched to the same finding when an analysis is re-run on the same file.

    Args:
        source: Deterministic or AI.
        check_id: Identifier of the check that produced the finding.
        issue_type: The kind of problem detected.
        column: Column name, or ``None`` for dataset-level findings.
        fingerprint: Extra discriminator when one check emits several findings
            for the same column (for example a specific business rule name).

    Returns:
        A 16 character hex digest.
    """
    raw = "|".join([source.value, check_id, issue_type.value, column or "*", fingerprint])
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


class Finding(BaseModel):
    """A single data quality problem detected in a dataset."""

    finding_id: str
    check_id: str
    issue_type: IssueType
    source: FindingSource = FindingSource.DETERMINISTIC
    severity: Severity = Severity.MEDIUM

    column: str | None = None
    title: str
    explanation: str

    affected_rows: int = 0
    row_count: int = 0
    row_indices: list[int] = Field(default_factory=list, max_length=MAX_ROW_SAMPLE)
    details: dict[str, Any] = Field(default_factory=dict)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)

    @property
    def affected_ratio(self) -> float:
        """Fraction of rows affected by this finding."""
        return self.affected_rows / self.row_count if self.row_count else 0.0

    @property
    def is_advisory(self) -> bool:
        """True when the finding must never drive an automatic change."""
        return self.source is FindingSource.AI


class FindingSet(BaseModel):
    """Container with convenience accessors over a list of findings."""

    findings: list[Finding] = Field(default_factory=list)

    def __len__(self) -> int:
        """Number of findings held."""
        return len(self.findings)

    def __iter__(self) -> Any:  # pragma: no cover - trivial delegation
        """Iterate over the findings."""
        return iter(self.findings)

    def by_source(self, source: FindingSource) -> list[Finding]:
        """Return findings produced by ``source``."""
        return [f for f in self.findings if f.source is source]

    def by_column(self, column: str) -> list[Finding]:
        """Return findings attached to ``column``."""
        return [f for f in self.findings if f.column == column]

    def by_severity(self) -> dict[Severity, int]:
        """Return a count of findings per severity level."""
        counts: dict[Severity, int] = dict.fromkeys(Severity, 0)
        for finding in self.findings:
            counts[finding.severity] += 1
        return counts

    def sorted(self) -> list[Finding]:
        """Return findings ordered by severity then by number of affected rows."""
        return sorted(
            self.findings,
            key=lambda f: (f.severity.rank, -f.affected_rows, f.column or ""),
        )
