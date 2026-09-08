"""Checks for missing values, including values that only pretend to be there."""

from __future__ import annotations

import re

import pandas as pd

from dqcopilot.models.enums import FindingSource, IssueType, Severity
from dqcopilot.models.findings import Finding, make_finding_id
from dqcopilot.profiling.type_inference import missing_mask, to_clean_strings
from dqcopilot.validation.base import Check, CheckContext, pct, ratio_severity, sample_indices
from dqcopilot.validation.registry import register_check

_NON_ALNUM = re.compile(r"[^a-z0-9]+")


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


#: Values that mean "no value" while occupying the cell. Stored squashed to letters and
#: digits, so "N/A", "n.a." and "n a" all collapse onto the same token.
PLACEHOLDER_TOKENS: frozenset[str] = frozenset(
    {
        "na",
        "nd",
        "nan",
        "nil",
        "none",
        "null",
        "tbd",
        "undefined",
        "unknown",
        "unspecified",
        "unavailable",
        "notavailable",
        "notapplicable",
        "notspecified",
        "nodata",
        "missing",
        "empty",
        "placeholder",
        "dummy",
        "test",
        "xxx",
        "xxxx",
        "sconosciuto",
        "ignoto",
        "nondisponibile",
        "nonspecificato",
        "nonapplicabile",
        "dadefinire",
        "mancante",
        "nessuno",
        "vuoto",
    }
)

#: Tokens no longer than this are ambiguous abbreviations ("ND" is also North Dakota),
#: so they are only trusted in columns that hold something longer than a short code.
_AMBIGUOUS_TOKEN_LENGTH = 3

#: Punctuation-only cells ("-", "--", "?") of at most this length read as "nothing here".
_DASH_MAX_LENGTH = 3


@register_check
class PlaceholderValueCheck(Check):
    """Report values that stand in for a missing one instead of being absent.

    A gap filled with the word "unknown" - or, in the dataset that prompted this check,
    with a legal form typed into a company-name field - is invisible to the missing-value
    check, because the cell is populated. It is still missing data, and every count of
    "complete rows" that trusts it is wrong.

    Two things are looked for. Known filler words are recognised anywhere. Beyond that, a
    value repeated many times inside a column whose values are otherwise nearly all
    distinct is treated as filler on structural grounds alone, which is what catches
    domain-specific placeholders that no word list could anticipate.
    """

    check_id = "placeholder_values"
    title = "Placeholder values"
    description = "Finds filler text used in place of a value that was never collected."

    #: Below this share of distinct values a column is a vocabulary rather than a list of
    #: names, and repetition says nothing about placeholders.
    MIN_UNIQUE_RATIO = 0.7
    #: A repeated filler has to appear at least this many times to be distinguishable
    #: from two records that genuinely share a name.
    MIN_REPEATS = 5
    #: ... and cover at least this share of the rows.
    MIN_REPEAT_RATIO = 0.01
    #: A value covering more than this share is the column's subject, not its filler.
    MAX_REPEAT_RATIO = 0.5

    def run(self, context: CheckContext) -> list[Finding]:  # noqa: D102 - inherited
        findings: list[Finding] = []
        row_count = context.row_count
        if row_count == 0:
            return findings

        for column in context.columns():
            series = context.series(column.name)
            if series.dtype != object and not pd.api.types.is_string_dtype(series):
                continue

            present = self._present_values(series)
            if present.empty:
                continue

            suspects = self._known_fillers(present) | self._repeated_fillers(present, row_count)
            if not suspects:
                continue

            mask = pd.Series(False, index=series.index)
            mask.loc[present.index] = present.str.casefold().isin(suspects).to_numpy()
            affected = int(mask.sum())
            if affected == 0:
                continue

            ratio = affected / row_count
            findings.append(
                Finding(
                    finding_id=make_finding_id(
                        FindingSource.DETERMINISTIC,
                        self.check_id,
                        IssueType.PLACEHOLDER_VALUE,
                        column.name,
                    ),
                    check_id=self.check_id,
                    issue_type=IssueType.PLACEHOLDER_VALUE,
                    severity=ratio_severity(ratio),
                    column=column.name,
                    title=f"Placeholder values in '{column.name}'",
                    explanation=(
                        f"{affected:,} row(s) in '{column.name}' ({pct(ratio)}) hold a value "
                        "that stands in for a missing one rather than being one: "
                        + "; ".join(repr(value) for value in sorted(suspects)[:5])
                        + ". These cells are counted as populated by every completeness "
                        "measure, so the column looks fuller than it is."
                    ),
                    affected_rows=affected,
                    row_count=row_count,
                    row_indices=sample_indices(mask),
                    details={"values": sorted(suspects)[:20]},
                )
            )
        return findings

    @staticmethod
    def _present_values(series: pd.Series) -> pd.Series:
        """Return the populated values of a column as stripped strings."""
        text = to_clean_strings(series)
        present = text[~missing_mask(series)].astype(str).str.strip()
        return present[present != ""]

    def _known_fillers(self, present: pd.Series) -> set[str]:
        """Return case-folded values that match the filler word list."""
        longest = int(present.str.len().max())
        found: set[str] = set()
        for value in present.unique():
            squashed = _squash(value)
            if squashed in PLACEHOLDER_TOKENS:
                # An ambiguous abbreviation counts as filler only in a column holding
                # more than short codes, otherwise every US state column reports "ND".
                if len(squashed) <= _AMBIGUOUS_TOKEN_LENGTH and longest <= _DASH_MAX_LENGTH:
                    continue
                found.add(str(value).casefold())
            elif not squashed and len(str(value)) <= _DASH_MAX_LENGTH:
                found.add(str(value).casefold())
        return found

    def _repeated_fillers(self, present: pd.Series, row_count: int) -> set[str]:
        """Return values repeated suspiciously often in an otherwise distinct column."""
        folded = present.str.casefold()
        if folded.nunique() / folded.size < self.MIN_UNIQUE_RATIO:
            return set()

        counts = folded.value_counts()
        floor = max(self.MIN_REPEATS, int(self.MIN_REPEAT_RATIO * row_count))
        repeated = counts[(counts >= floor) & (counts / row_count <= self.MAX_REPEAT_RATIO)]
        return {str(value) for value in repeated.index}


def _squash(value: object) -> str:
    """Reduce a value to its lowercase letters and digits."""
    return _NON_ALNUM.sub("", str(value).casefold())
