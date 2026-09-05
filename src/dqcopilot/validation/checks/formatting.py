"""Checks for cosmetic text problems: stray whitespace and inconsistent capitalisation."""

from __future__ import annotations

import pandas as pd

from dqcopilot.models.enums import FindingSource, IssueType, Severity
from dqcopilot.models.findings import Finding, make_finding_id
from dqcopilot.models.profile import ColumnProfile
from dqcopilot.profiling.type_inference import to_clean_strings
from dqcopilot.validation.base import Check, CheckContext, pct, sample_indices
from dqcopilot.validation.registry import register_check

_TEXTUAL_DTYPES = ("object", "string")


def _is_textual(series: pd.Series) -> bool:
    """True when a column holds strings rather than native numbers or dates."""
    return series.dtype == object or pd.api.types.is_string_dtype(series)


@register_check
class WhitespaceCheck(Check):
    """Report values with leading or trailing whitespace."""

    check_id = "leading_trailing_whitespace"
    title = "Leading or trailing spaces"
    description = "Finds text values padded with spaces, tabs or non-breaking spaces."

    def run(self, context: CheckContext) -> list[Finding]:  # noqa: D102 - inherited
        findings: list[Finding] = []
        row_count = context.row_count

        for column in context.columns():
            series = context.series(column.name)
            if not _is_textual(series):
                continue

            text = to_clean_strings(series)
            padded = (text.str.strip() != text) & text.notna() & (text.str.strip() != "")
            affected = int(padded.sum())
            if affected == 0:
                continue

            ratio = affected / row_count if row_count else 0.0
            findings.append(
                Finding(
                    finding_id=make_finding_id(
                        FindingSource.DETERMINISTIC,
                        self.check_id,
                        IssueType.LEADING_TRAILING_WHITESPACE,
                        column.name,
                    ),
                    check_id=self.check_id,
                    issue_type=IssueType.LEADING_TRAILING_WHITESPACE,
                    severity=Severity.LOW,
                    column=column.name,
                    title=f"Padded values in '{column.name}'",
                    explanation=(
                        f"{affected:,} values in '{column.name}' ({pct(ratio)}) start or end "
                        "with whitespace. Padding is invisible on screen but makes joins, "
                        "grouping and duplicate detection fail."
                    ),
                    affected_rows=affected,
                    row_count=row_count,
                    row_indices=sample_indices(padded),
                    details={"examples": _padded_examples(text, padded)},
                )
            )
        return findings


@register_check
class CapitalizationCheck(Check):
    """Report categorical columns whose values differ only by letter case."""

    check_id = "inconsistent_capitalization"
    title = "Inconsistent capitalisation"
    description = "Finds values that are the same word written with different casing."

    #: Only columns with at most this many distinct values are considered categorical.
    MAX_LEVELS = 200

    def run(self, context: CheckContext) -> list[Finding]:  # noqa: D102 - inherited
        findings: list[Finding] = []
        row_count = context.row_count

        for column in context.columns():
            series = context.series(column.name)
            if not _is_textual(series) or not self._is_candidate(column):
                continue

            text = to_clean_strings(series).str.strip()
            present = text[text.notna() & (text != "")]
            if present.empty:
                continue

            folded = present.str.casefold()
            variants = self._variants(present, folded)
            if not variants:
                continue

            affected_mask = pd.Series(False, index=series.index)
            affected_mask.loc[present.index] = folded.isin(variants.keys()).to_numpy()
            affected = int(affected_mask.sum())

            findings.append(
                Finding(
                    finding_id=make_finding_id(
                        FindingSource.DETERMINISTIC,
                        self.check_id,
                        IssueType.INCONSISTENT_CAPITALIZATION,
                        column.name,
                    ),
                    check_id=self.check_id,
                    issue_type=IssueType.INCONSISTENT_CAPITALIZATION,
                    severity=Severity.LOW,
                    column=column.name,
                    title=f"Mixed capitalisation in '{column.name}'",
                    explanation=(
                        f"{len(variants)} value(s) in '{column.name}' appear with more than "
                        f"one capitalisation, affecting {affected:,} rows "
                        f"({pct(affected / row_count if row_count else 0)}). For example: "
                        + "; ".join(
                            f"{' / '.join(repr(v) for v in sorted(spellings)[:4])}"
                            for spellings in list(variants.values())[:3]
                        )
                        + "."
                    ),
                    affected_rows=affected,
                    row_count=row_count,
                    row_indices=sample_indices(affected_mask),
                    details={
                        "variant_groups": {
                            key: sorted(spellings) for key, spellings in list(variants.items())[:20]
                        }
                    },
                )
            )
        return findings

    def _is_candidate(self, column: ColumnProfile) -> bool:
        return 0 < column.unique_count <= self.MAX_LEVELS

    @staticmethod
    def _variants(present: pd.Series, folded: pd.Series) -> dict[str, set[str]]:
        """Return case-folded keys that map to more than one written form."""
        groups: dict[str, set[str]] = {}
        for original, key in zip(present, folded, strict=True):
            groups.setdefault(str(key), set()).add(str(original))
        return {key: spellings for key, spellings in groups.items() if len(spellings) > 1}


def _padded_examples(text: pd.Series, mask: pd.Series, limit: int = 5) -> list[str]:
    """Return a few padded values rendered with visible delimiters."""
    return [f"[{value}]" for value in text[mask].head(limit).tolist()]
