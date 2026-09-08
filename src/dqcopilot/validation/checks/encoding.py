"""Check for text damaged by being decoded with the wrong codec."""

from __future__ import annotations

from collections.abc import Hashable

import pandas as pd

from dqcopilot.encoding_repair import has_lost_characters, repair_mojibake
from dqcopilot.models.enums import FindingSource, IssueType, Severity
from dqcopilot.models.findings import Finding, make_finding_id
from dqcopilot.profiling.type_inference import to_clean_strings
from dqcopilot.validation.base import Check, CheckContext, pct, ratio_severity, sample_indices
from dqcopilot.validation.registry import register_check


def _is_textual(series: pd.Series) -> bool:
    """True when a column holds strings rather than native numbers or dates."""
    return series.dtype == object or pd.api.types.is_string_dtype(series)


@register_check
class CorruptedEncodingCheck(Check):
    """Report values that were decoded with the wrong codec.

    This is the defect that survives every other check: the file parses, the column
    profiles as text, and nothing looks wrong until a human reads a row. It is reported
    at least at medium severity even when it touches a handful of rows, because the
    damage is silent and spreads into every export made from the dataset.
    """

    check_id = "corrupted_encoding"
    title = "Corrupted text encoding"
    description = "Finds text that was read as cp1252 or latin-1 when it was really UTF-8."

    def run(self, context: CheckContext) -> list[Finding]:  # noqa: D102 - inherited
        findings: list[Finding] = []
        row_count = context.row_count
        if row_count == 0:
            return findings

        for column in context.columns():
            series = context.series(column.name)
            if not _is_textual(series):
                continue

            text = to_clean_strings(series)
            repairs, lost = self._scan(text)
            affected_mask = pd.Series(
                text.index.isin([index for index, _, _ in repairs] + lost),
                index=text.index,
            )
            affected = int(affected_mask.sum())
            if affected == 0:
                continue

            ratio = affected / row_count
            findings.append(
                Finding(
                    finding_id=make_finding_id(
                        FindingSource.DETERMINISTIC,
                        self.check_id,
                        IssueType.CORRUPTED_ENCODING,
                        column.name,
                    ),
                    check_id=self.check_id,
                    issue_type=IssueType.CORRUPTED_ENCODING,
                    severity=self._severity(ratio),
                    column=column.name,
                    title=f"Corrupted characters in '{column.name}'",
                    explanation=self._explain(column.name, repairs, lost, ratio),
                    affected_rows=affected,
                    row_count=row_count,
                    row_indices=sample_indices(affected_mask),
                    details={
                        "examples": [
                            f"{damaged!r} -> {repaired!r}" for _, damaged, repaired in repairs[:5]
                        ],
                        "unrecoverable": len(lost),
                    },
                )
            )
        return findings

    @staticmethod
    def _scan(text: pd.Series) -> tuple[list[tuple[Hashable, str, str]], list[Hashable]]:
        """Return (index, damaged, repaired) triples, and the indices beyond repair."""
        repairs: list[tuple[Hashable, str, str]] = []
        lost: list[Hashable] = []
        for index, value in text.items():
            if not isinstance(value, str):
                continue
            if has_lost_characters(value):
                lost.append(index)
                continue
            repaired = repair_mojibake(value)
            if repaired is not None:
                repairs.append((index, value, repaired))
        return repairs, lost

    @staticmethod
    def _severity(ratio: float) -> Severity:
        """Escalate with the affected share, but never report below medium.

        A ratio-only rule would file three corrupted names out of five hundred as
        cosmetic. The share tells you how much data is affected; it says nothing about
        how bad the defect is, and silent corruption is bad at any share.
        """
        severity = ratio_severity(ratio)
        return Severity.MEDIUM if severity is Severity.LOW else severity

    def _explain(
        self,
        column: str,
        repairs: list[tuple[Hashable, str, str]],
        lost: list[Hashable],
        ratio: float,
    ) -> str:
        parts = [
            f"{len(repairs) + len(lost):,} value(s) in '{column}' ({pct(ratio)}) contain "
            "characters that were decoded with the wrong codec: the file is UTF-8 but was "
            "read as cp1252 or latin-1 somewhere upstream."
        ]
        if repairs:
            parts.append(
                f"{len(repairs):,} of them can be decoded back to the original text, "
                "because the bytes survived intact."
            )
        if lost:
            parts.append(
                f"{len(lost):,} contain the replacement character (U+FFFD), which means a "
                "byte was already discarded. That text cannot be recovered from this file "
                "and has to be re-exported from the source."
            )
        return " ".join(parts)
