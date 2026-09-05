"""Checks for values and schemas that disagree with themselves."""

from __future__ import annotations

import re
from collections import defaultdict

import pandas as pd

from dqcopilot.models.enums import FindingSource, IssueType, SemanticType, Severity
from dqcopilot.models.findings import Finding, make_finding_id
from dqcopilot.profiling.type_inference import (
    coerce_numeric,
    match_date_formats,
    missing_mask,
    parse_boolean_token,
    to_clean_strings,
)
from dqcopilot.validation.base import Check, CheckContext, pct, sample_indices
from dqcopilot.validation.registry import register_check

#: Columns with more distinct values than this are treated as free text, not categories.
MAX_CATEGORY_LEVELS = 100
#: Below this share of rows, a category level is considered a probable variant of another.
RARE_LEVEL_RATIO = 0.02

_NON_ALPHANUMERIC = re.compile(r"[^a-z0-9]+")


def _canonical(value: str) -> str:
    """Reduce a category value to a comparison key (lowercase, alphanumeric only)."""
    return _NON_ALPHANUMERIC.sub("", value.strip().casefold())


@register_check
class InconsistentCategoryCheck(Check):
    """Report category values that are probably the same thing written differently.

    Grouping is deterministic: values collapse to the same key when they are identical
    after lower-casing and removing every non-alphanumeric character. That catches
    ``"United Kingdom"`` / ``"united-kingdom"`` / ``"UNITED KINGDOM"`` without any fuzzy
    matching. Genuinely different spellings (``"UK"`` vs ``"United Kingdom"``) are left
    to the AI suggestion panel, where a human decides.
    """

    check_id = "inconsistent_category"
    title = "Inconsistent category values"
    description = "Finds category values that differ only by punctuation or spacing."

    def run(self, context: CheckContext) -> list[Finding]:  # noqa: D102 - inherited
        findings: list[Finding] = []

        for column in context.columns():
            if column.semantic_type not in (SemanticType.CATEGORICAL, SemanticType.BOOLEAN):
                continue
            if column.unique_count > MAX_CATEGORY_LEVELS:
                continue

            series = context.series(column.name)
            present = to_clean_strings(series)[~missing_mask(series)]
            if present.empty:
                continue

            trimmed = present.str.strip()
            groups: dict[str, set[str]] = defaultdict(set)
            for value in trimmed.unique():
                groups[_canonical(str(value))].add(str(value))

            variants = {key: forms for key, forms in groups.items() if len(forms) > 1}
            # Case-only differences are already reported by the capitalisation check.
            variants = {
                key: forms
                for key, forms in variants.items()
                if len({form.casefold() for form in forms}) > 1
            }
            if not variants:
                continue

            affected_values = {form for forms in variants.values() for form in forms}
            mask = pd.Series(False, index=series.index)
            mask.loc[trimmed.index] = trimmed.isin(affected_values).to_numpy()
            affected = int(mask.sum())

            findings.append(
                Finding(
                    finding_id=make_finding_id(
                        FindingSource.DETERMINISTIC,
                        self.check_id,
                        IssueType.INCONSISTENT_CATEGORY,
                        column.name,
                    ),
                    check_id=self.check_id,
                    issue_type=IssueType.INCONSISTENT_CATEGORY,
                    severity=Severity.MEDIUM,
                    column=column.name,
                    title=f"Inconsistent categories in '{column.name}'",
                    explanation=(
                        f"{len(variants)} group(s) of values in '{column.name}' differ only "
                        f"by spacing or punctuation, affecting {affected:,} rows "
                        f"({pct(affected / context.row_count if context.row_count else 0)}). "
                        "Grouping or joining on this column will split what should be one "
                        "category. Examples: "
                        + "; ".join(
                            " / ".join(repr(form) for form in sorted(forms))
                            for forms in list(variants.values())[:3]
                        )
                        + "."
                    ),
                    affected_rows=affected,
                    row_count=context.row_count,
                    row_indices=sample_indices(mask),
                    details={
                        "variant_groups": {
                            key: sorted(forms) for key, forms in list(variants.items())[:20]
                        }
                    },
                )
            )
        return findings


@register_check
class MixedDatatypeCheck(Check):
    """Report columns that mix genuinely different kinds of value.

    A column is 'mixed' when it holds a substantial share of two different kinds - for
    example 70% numbers and 30% words. Small minorities are left to the more specific
    checks (invalid dates, invalid emails), which explain the problem better.
    """

    check_id = "mixed_datatypes"
    title = "Mixed data types"
    description = "Finds columns that mix numbers, dates and text in the same field."

    #: Both kinds must reach this share before the column counts as mixed.
    MINORITY_THRESHOLD = 0.1

    def run(self, context: CheckContext) -> list[Finding]:  # noqa: D102 - inherited
        findings: list[Finding] = []

        for column in context.columns():
            series = context.series(column.name)
            if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_datetime64_any_dtype(
                series
            ):
                continue
            if column.semantic_type in (SemanticType.EMPTY, SemanticType.IDENTIFIER):
                continue

            present = series[~missing_mask(series)]
            total = int(present.size)
            if total < 5:
                continue

            shares = self._kind_shares(present, total)
            significant = {
                kind: share for kind, share in shares.items() if share >= self.MINORITY_THRESHOLD
            }
            if len(significant) < 2:
                continue

            breakdown = ", ".join(
                f"{share:.0%} {kind}"
                for kind, share in sorted(significant.items(), key=lambda i: -i[1])
            )

            findings.append(
                Finding(
                    finding_id=make_finding_id(
                        FindingSource.DETERMINISTIC,
                        self.check_id,
                        IssueType.MIXED_DATATYPES,
                        column.name,
                    ),
                    check_id=self.check_id,
                    issue_type=IssueType.MIXED_DATATYPES,
                    severity=Severity.MEDIUM,
                    column=column.name,
                    title=f"'{column.name}' mixes data types",
                    explanation=(
                        f"'{column.name}' contains more than one kind of value: {breakdown}. "
                        "A column that mixes types cannot be loaded into a typed warehouse "
                        "column without losing rows, and aggregations over it are unreliable."
                    ),
                    affected_rows=total,
                    row_count=context.row_count,
                    details={"shares": {kind: round(share, 4) for kind, share in shares.items()}},
                    confidence=0.85,
                )
            )
        return findings

    @staticmethod
    def _kind_shares(present: pd.Series, total: int) -> dict[str, float]:
        """Classify every value as number/date/boolean/text and return the shares."""
        text = to_clean_strings(present).str.strip()

        numeric = coerce_numeric(present).notna()
        dates = text.map(lambda value: bool(match_date_formats(str(value))))
        booleans = text.map(lambda value: parse_boolean_token(str(value)) is not None)

        # Assign each value to exactly one kind, most specific first.
        is_number = numeric.fillna(False).astype(bool)
        is_date = dates.astype(bool) & ~is_number
        is_boolean = booleans.fillna(False).astype(bool) & ~is_number & ~is_date
        is_text = ~(is_number | is_date | is_boolean)

        return {
            "numbers": float(is_number.sum()) / total,
            "dates": float(is_date.sum()) / total,
            "booleans": float(is_boolean.sum()) / total,
            "text": float(is_text.sum()) / total,
        }


@register_check
class SchemaCheck(Check):
    """Report structural problems with the file's header row."""

    check_id = "schema_mismatch"
    title = "Schema problems"
    description = "Reports headers that were empty, duplicated or padded with whitespace."

    def run(self, context: CheckContext) -> list[Finding]:  # noqa: D102 - inherited
        if not context.notes:
            return []

        return [
            Finding(
                finding_id=make_finding_id(
                    FindingSource.DETERMINISTIC,
                    self.check_id,
                    IssueType.SCHEMA_MISMATCH,
                    None,
                ),
                check_id=self.check_id,
                issue_type=IssueType.SCHEMA_MISMATCH,
                severity=Severity.MEDIUM,
                column=None,
                title="The header row needed repairs",
                explanation=(
                    f"{len(context.notes)} problem(s) were found in the header row and "
                    "repaired while reading the file so the analysis could continue: "
                    + " ".join(context.notes[:6])
                    + " Fix these at the source, because the repaired names are guesses."
                ),
                affected_rows=0,
                row_count=context.row_count,
                details={"notes": context.notes[:20]},
            )
        ]
