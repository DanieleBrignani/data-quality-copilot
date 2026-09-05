"""The data quality score.

The formula is deliberately simple, fully deterministic and documented in the README.
It is a **presentation aid for this demo, not a validated data quality metric**: it has
no academic backing and the weights were chosen by hand.

Definition
----------
Every finding carries a severity weight and an affected-row ratio::

    weight(critical) = 1.0   weight(high) = 0.6   weight(medium) = 0.3
    weight(low)      = 0.1   weight(info) = 0.0

    penalty(finding) = weight(severity) x affected_ratio x 100

For each column::

    column_score = clamp(100 - sum(penalty of that column's findings), 0, 100)

For the dataset::

    overall_score = clamp(mean(column_scores) - sum(penalty of dataset-level findings), 0, 100)

Only deterministic findings contribute. AI suggestions are advisory and never move the
score, so the number stays reproducible without an API key.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from dqcopilot.models.enums import FindingSource, Severity
from dqcopilot.models.findings import FindingSet
from dqcopilot.models.profile import DatasetProfile

MAX_SCORE = 100.0


class ColumnScore(BaseModel):
    """Quality score for a single column."""

    column: str
    score: float = Field(ge=0.0, le=100.0)
    penalty: float = Field(ge=0.0)
    finding_count: int = 0


class QualityScore(BaseModel):
    """Overall quality score plus its per-column breakdown."""

    overall: float = Field(ge=0.0, le=100.0)
    columns: list[ColumnScore] = Field(default_factory=list)
    dataset_penalty: float = Field(default=0.0, ge=0.0)
    severity_counts: dict[str, int] = Field(default_factory=dict)
    #: Human readable formula, embedded in the report so the number is never a black box.
    formula: str = (
        "column_score = 100 - SUM(severity_weight x affected_ratio x 100); "
        "overall = mean(column_scores) - dataset_level_penalties; clamped to [0, 100]"
    )

    @property
    def grade(self) -> str:
        """A coarse letter grade, used only for display."""
        if self.overall >= 90:
            return "A"
        if self.overall >= 75:
            return "B"
        if self.overall >= 60:
            return "C"
        if self.overall >= 40:
            return "D"
        return "E"

    def column_score(self, name: str) -> float:
        """Return the score of ``name``, or 100 when the column is unknown."""
        for entry in self.columns:
            if entry.column == name:
                return entry.score
        return MAX_SCORE


def _clamp(value: float) -> float:
    return max(0.0, min(MAX_SCORE, value))


def compute_quality_score(profile: DatasetProfile, findings: FindingSet) -> QualityScore:
    """Compute the deterministic quality score for an analysis.

    Args:
        profile: The dataset profile (used for the column list).
        findings: All findings; AI-sourced findings are ignored on purpose.

    Returns:
        A :class:`QualityScore`.
    """
    deterministic = [f for f in findings.findings if f.source is FindingSource.DETERMINISTIC]

    penalties: dict[str, float] = {column.name: 0.0 for column in profile.columns}
    counts: dict[str, int] = {column.name: 0 for column in profile.columns}
    dataset_penalty = 0.0

    for finding in deterministic:
        penalty = finding.severity.weight * finding.affected_ratio * MAX_SCORE
        if finding.column is None or finding.column not in penalties:
            dataset_penalty += penalty
        else:
            penalties[finding.column] += penalty
            counts[finding.column] += 1

    column_scores = [
        ColumnScore(
            column=name,
            score=round(_clamp(MAX_SCORE - penalty), 2),
            penalty=round(penalty, 2),
            finding_count=counts[name],
        )
        for name, penalty in penalties.items()
    ]

    mean_column_score = (
        sum(entry.score for entry in column_scores) / len(column_scores)
        if column_scores
        else MAX_SCORE
    )
    overall = _clamp(mean_column_score - dataset_penalty)

    severity_counts = {severity.value: 0 for severity in Severity}
    for finding in deterministic:
        severity_counts[finding.severity.value] += 1

    return QualityScore(
        overall=round(overall, 2),
        columns=column_scores,
        dataset_penalty=round(dataset_penalty, 2),
        severity_counts=severity_counts,
    )
