"""Checks for duplicated rows."""

from __future__ import annotations

import pandas as pd

from dqcopilot.models.enums import FindingSource, IssueType, Severity
from dqcopilot.models.findings import Finding, make_finding_id
from dqcopilot.validation.base import Check, CheckContext, name_matches, pct, sample_indices
from dqcopilot.validation.registry import register_check


@register_check
class ExactDuplicateRowsCheck(Check):
    """Report rows that are byte-for-byte identical to an earlier row."""

    check_id = "exact_duplicate_rows"
    title = "Exact duplicate rows"
    description = "Finds rows whose values are identical across every column."

    def run(self, context: CheckContext) -> list[Finding]:  # noqa: D102 - inherited
        row_count = context.row_count
        if row_count < 2:
            return []

        normalised = context.normalised_frame()
        mask = normalised.duplicated(keep="first")
        affected = int(mask.sum())
        if affected == 0:
            return []

        ratio = affected / row_count
        severity = Severity.HIGH if ratio > 0.05 else Severity.MEDIUM
        group_count = int(
            normalised[normalised.duplicated(keep=False)]
            .groupby(list(normalised.columns), dropna=False)
            .ngroups
        )

        return [
            Finding(
                finding_id=make_finding_id(
                    FindingSource.DETERMINISTIC,
                    self.check_id,
                    IssueType.EXACT_DUPLICATE_ROWS,
                    None,
                ),
                check_id=self.check_id,
                issue_type=IssueType.EXACT_DUPLICATE_ROWS,
                severity=severity,
                column=None,
                title="Exact duplicate rows",
                explanation=(
                    f"{affected:,} of {row_count:,} rows ({pct(ratio)}) repeat a row that "
                    f"already appears earlier in the file, across {group_count:,} distinct "
                    "group(s). Values are compared after trimming surrounding whitespace "
                    "and ignoring letter case."
                ),
                affected_rows=affected,
                row_count=row_count,
                row_indices=sample_indices(mask),
                details={"duplicate_groups": group_count, "comparison": "trimmed, case-folded"},
            )
        ]


@register_check
class ProbableDuplicateRowsCheck(Check):
    """Report rows that are probably the same record written slightly differently.

    The comparison is deterministic: a *blocking key* is built from the identifying
    columns (names, emails, identifiers) after aggressive normalisation - lower-casing,
    removing punctuation, collapsing whitespace and stripping common company suffixes
    such as ``Ltd``, ``GmbH`` or ``S.p.A.``. Rows that collide on that key but are not
    exact duplicates are reported for human review.

    This is intentionally not fuzzy string matching: an exact match on a normalised key
    is explainable and reproducible, which matters more here than recall.
    """

    check_id = "probable_duplicate_rows"
    title = "Probable duplicate rows"
    description = "Finds near-identical records that differ only by formatting."

    #: Legal-form tokens removed before comparing organisation names.
    COMPANY_SUFFIXES = (
        "ltd",
        "limited",
        "llc",
        "inc",
        "incorporated",
        "corp",
        "corporation",
        "gmbh",
        "ag",
        "bv",
        "nv",
        "sa",
        "sas",
        "sarl",
        "srl",
        "spa",
        "plc",
        "co",
        "company",
        "group",
        "holding",
        "holdings",
    )

    #: Column names that describe *which entity* a row is about.
    KEY_HINTS = (
        "name",
        "company",
        "supplier",
        "vendor",
        "customer",
        "client",
        "organisation",
        "organization",
        "email",
        "vat",
        "siren",
    )

    #: Tokens marking a surrogate key. These are deliberately excluded from the blocking
    #: key: a record re-entered by mistake gets a *new* identifier, so including the id
    #: would guarantee the two rows never collide - defeating the whole check.
    SURROGATE_KEY_TOKENS = ("id", "ids", "uuid", "guid", "pk", "key", "row")

    def run(self, context: CheckContext) -> list[Finding]:  # noqa: D102 - inherited
        row_count = context.row_count
        if row_count < 2:
            return []

        key_columns = self._key_columns(context)
        if not key_columns:
            return []

        key = self._blocking_key(context.frame, key_columns)
        usable = key.str.len() > 0
        exact = context.normalised_frame().duplicated(keep=False)

        collides = key.duplicated(keep=False) & usable
        mask = collides & ~exact
        affected = int(mask.sum())
        if affected == 0:
            return []

        ratio = affected / row_count
        groups = int(key[mask].nunique())
        examples = self._examples(context, key, mask, key_columns)

        return [
            Finding(
                finding_id=make_finding_id(
                    FindingSource.DETERMINISTIC,
                    self.check_id,
                    IssueType.PROBABLE_DUPLICATE_ROWS,
                    None,
                ),
                check_id=self.check_id,
                issue_type=IssueType.PROBABLE_DUPLICATE_ROWS,
                severity=Severity.HIGH if ratio > 0.02 else Severity.MEDIUM,
                column=None,
                title="Probable duplicate records",
                explanation=(
                    f"{affected:,} rows ({pct(ratio)}) share an identity key with another "
                    f"row without being exact duplicates, forming {groups:,} group(s). The "
                    f"key is built from {', '.join(repr(c) for c in key_columns)} after "
                    "lower-casing, removing punctuation and dropping company legal forms. "
                    "These need a human decision: merging records is not reversible."
                ),
                affected_rows=affected,
                row_count=row_count,
                row_indices=sample_indices(mask),
                details={
                    "key_columns": key_columns,
                    "duplicate_groups": groups,
                    "examples": examples,
                },
                confidence=0.8,
            )
        ]

    # ------------------------------------------------------------------ internals

    def _key_columns(self, context: CheckContext) -> list[str]:
        """Pick the columns that describe which entity a row is about.

        Returns an empty list when the dataset has no entity-describing column - a
        transaction table keyed only by surrogate ids gets no near-duplicate findings,
        which is correct: repeated customer ids there are legitimate, not duplicates.
        """
        return [
            column.name
            for column in context.columns()
            if name_matches(column.name, self.KEY_HINTS)
            and not name_matches(column.name, self.SURROGATE_KEY_TOKENS)
        ][:3]

    def _blocking_key(self, frame: pd.DataFrame, columns: list[str]) -> pd.Series:
        parts = [self._normalise_key(frame[column]) for column in columns]
        key = parts[0]
        for part in parts[1:]:
            key = key.str.cat(part, sep="|")
        return key

    def _normalise_key(self, series: pd.Series) -> pd.Series:
        """Reduce a value to a comparison key.

        Dots are dropped from ordinary values (so ``S.p.A.`` and ``SpA`` collide) but
        kept inside anything that looks like an email address, where they are part of
        the domain.
        """
        text = series.astype("string").fillna("").str.lower()
        text = text.str.replace(r"[^\w\s@.]+", " ", regex=True)

        has_at = text.str.contains("@", regex=False)
        text = text.where(has_at, text.str.replace(".", "", regex=False))

        suffix_pattern = r"\b(?:" + "|".join(self.COMPANY_SUFFIXES) + r")\b"
        text = text.str.replace(suffix_pattern, " ", regex=True)
        text = text.str.replace(r"\s+", " ", regex=True).str.strip()
        return text.astype(str)

    def _examples(
        self,
        context: CheckContext,
        key: pd.Series,
        mask: pd.Series,
        key_columns: list[str],
    ) -> list[dict[str, object]]:
        """Return a few readable duplicate groups for the UI."""
        examples: list[dict[str, object]] = []
        for group_key in key[mask].unique()[:5]:
            positions = key.index[key == group_key].tolist()[:4]
            examples.append(
                {
                    "rows": [int(position) for position in positions],
                    "values": [
                        {column: str(context.frame.at[position, column]) for column in key_columns}
                        for position in positions
                    ],
                }
            )
        return examples
