"""Checks for missing values."""

from __future__ import annotations

from dqcopilot.models.enums import FindingSource, IssueType, Severity
from dqcopilot.models.findings import Finding, make_finding_id
from dqcopilot.profiling.type_inference import missing_mask
from dqcopilot.validation.base import Check, CheckContext, pct, ratio_severity, sample_indices
from dqcopilot.validation.registry import register_check


@register_check
class MissingValuesCheck(Check):
    """Report columns that contain missing values.

    A value is missing when it is null **or** a string of only whitespace, matching the
    definition used by the profiler.
    """

    check_id = "missing_values"
    title = "Missing values"
    description = "Finds empty, null and whitespace-only cells in each column."

    def run(self, context: CheckContext) -> list[Finding]:  # noqa: D102 - inherited
        findings: list[Finding] = []
        row_count = context.row_count
        if row_count == 0:
            return findings

        for column in context.columns():
            mask = missing_mask(context.series(column.name))
            affected = int(mask.sum())
            if affected == 0:
                continue

            ratio = affected / row_count
            severity = (
                Severity.CRITICAL
                if ratio >= 1.0
                else ratio_severity(ratio, critical_above=0.5, high_above=0.2, medium_above=0.02)
            )
            explanation = (
                f"{affected:,} of {row_count:,} rows ({pct(ratio)}) have no value in "
                f"'{column.name}'."
            )
            if ratio >= 1.0:
                explanation += " The column is entirely empty and carries no information."

            findings.append(
                Finding(
                    finding_id=make_finding_id(
                        FindingSource.DETERMINISTIC,
                        self.check_id,
                        IssueType.MISSING_VALUES,
                        column.name,
                    ),
                    check_id=self.check_id,
                    issue_type=IssueType.MISSING_VALUES,
                    severity=severity,
                    column=column.name,
                    title=f"Missing values in '{column.name}'",
                    explanation=explanation,
                    affected_rows=affected,
                    row_count=row_count,
                    row_indices=sample_indices(mask),
                    details={
                        "missing_ratio": round(ratio, 4),
                        "semantic_type": column.semantic_type.value,
                    },
                )
            )
        return findings


@register_check
class ConstantColumnCheck(Check):
    """Report columns that hold a single repeated value."""

    check_id = "constant_column"
    title = "Constant column"
    description = "Finds columns where every populated row holds the same value."

    def run(self, context: CheckContext) -> list[Finding]:  # noqa: D102 - inherited
        findings: list[Finding] = []
        if context.row_count < 2:
            return findings

        for column in context.columns():
            if column.unique_count != 1 or column.missing_count == column.row_count:
                continue
            value = column.top_values[0].value if column.top_values else "(unknown)"
            findings.append(
                Finding(
                    finding_id=make_finding_id(
                        FindingSource.DETERMINISTIC,
                        self.check_id,
                        IssueType.CONSTANT_COLUMN,
                        column.name,
                    ),
                    check_id=self.check_id,
                    issue_type=IssueType.CONSTANT_COLUMN,
                    severity=Severity.INFO,
                    column=column.name,
                    title=f"'{column.name}' is constant",
                    explanation=(
                        f"Every populated row of '{column.name}' holds the same value "
                        f"('{value}'). The column adds no discriminating information; "
                        "this is worth confirming rather than fixing automatically."
                    ),
                    affected_rows=context.row_count - column.missing_count,
                    row_count=context.row_count,
                    details={"constant_value": value},
                )
            )
        return findings
