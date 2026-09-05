"""Build the minimal payload sent to the Anthropic API.

This module is the privacy boundary of the application. **The dataset never leaves the
machine.** What is sent is a compact description of its *shape*: column names, inferred
types, missing counts, distinct counts, and - only where it is needed to answer the
question - a small number of example values.

Three rules keep the sample values defensible:

1. Columns whose inferred type marks them as personal or identifying (emails,
   identifiers) are never sampled verbatim. A *shape mask* is sent instead, so
   ``alice.martin@example.com`` becomes ``aaaaa.aaaaaa@aaaaaaa.aaa``, which is enough to
   reason about format and nothing else.
2. Every sampled value is truncated, and the number of values per column and columns per
   request are both capped by configuration.
3. Sampling can be switched off entirely with ``AI_SEND_SAMPLES=false``, in which case
   only counts and types are sent.
"""

from __future__ import annotations

import re
from typing import Any

import pandas as pd

from dqcopilot.config import Settings
from dqcopilot.models.enums import SemanticType
from dqcopilot.models.findings import Finding
from dqcopilot.models.profile import ColumnProfile, DatasetProfile
from dqcopilot.profiling.type_inference import missing_mask, to_clean_strings

#: Inferred types that must never be sent as raw values.
NEVER_SAMPLE_VERBATIM: frozenset[SemanticType] = frozenset(
    {SemanticType.EMAIL, SemanticType.IDENTIFIER}
)

#: Column names that force masking regardless of the inferred type.
SENSITIVE_NAME_TOKENS: tuple[str, ...] = (
    "name",
    "email",
    "phone",
    "mobile",
    "address",
    "street",
    "city",
    "postcode",
    "zip",
    "iban",
    "bic",
    "vat",
    "ssn",
    "nino",
    "passport",
    "birth",
    "dob",
    "salary",
    "password",
    "token",
    "secret",
)

MAX_VALUE_LENGTH = 40

_DIGIT = re.compile(r"\d")
_LETTER = re.compile(r"[^\W\d_]", re.UNICODE)


def mask_value(value: str, max_length: int = MAX_VALUE_LENGTH) -> str:
    """Replace letters with ``a`` and digits with ``9``, keeping punctuation.

    The result describes the *format* of a value without disclosing its content, which
    is exactly what the model needs to reason about, say, a malformed VAT number.
    """
    truncated = value[:max_length]
    masked = _DIGIT.sub("9", truncated)
    masked = _LETTER.sub("a", masked)
    return masked + ("…" if len(value) > max_length else "")


def should_mask(column: ColumnProfile) -> bool:
    """True when a column's values must be masked rather than sent verbatim."""
    if column.semantic_type in NEVER_SAMPLE_VERBATIM:
        return True
    lowered = column.name.lower()
    return any(token in lowered for token in SENSITIVE_NAME_TOKENS)


def column_summary(
    column: ColumnProfile,
    series: pd.Series | None,
    settings: Settings,
) -> dict[str, Any]:
    """Describe one column for the model."""
    summary: dict[str, Any] = {
        "name": column.name,
        "inferred_type": column.semantic_type.value,
        "missing": column.missing_count,
        "rows": column.row_count,
        "distinct": column.unique_count,
    }
    if column.numeric is not None and column.numeric.minimum is not None:
        summary["min"] = round(column.numeric.minimum, 4)
        summary["max"] = round(column.numeric.maximum or 0.0, 4)
    if column.temporal is not None:
        summary["earliest"] = column.temporal.minimum
        summary["latest"] = column.temporal.maximum

    if not settings.ai_send_samples or series is None or settings.ai_sample_rows == 0:
        return summary

    masked = should_mask(column)
    samples = _sample_values(series, settings.ai_sample_rows)
    summary["examples"] = [
        mask_value(value) if masked else value[:MAX_VALUE_LENGTH] for value in samples
    ]
    summary["examples_are_masked"] = masked
    return summary


def _sample_values(series: pd.Series, limit: int) -> list[str]:
    """Return up to ``limit`` distinct non-missing values, as strings."""
    present = to_clean_strings(series)[~missing_mask(series)]
    if present.empty:
        return []
    return [str(value) for value in present.drop_duplicates().head(limit)]


def dataset_payload(
    profile: DatasetProfile,
    frame: pd.DataFrame | None,
    settings: Settings,
) -> dict[str, Any]:
    """Build the shape-only description of a dataset."""
    columns = profile.columns[: settings.ai_max_columns]
    return {
        "source_name": profile.source_name,
        "row_count": profile.row_count,
        "column_count": profile.column_count,
        "columns_included": len(columns),
        "columns": [
            column_summary(
                column,
                frame[column.name] if frame is not None and column.name in frame else None,
                settings,
            )
            for column in columns
        ],
    }


def category_payload(
    column: ColumnProfile,
    series: pd.Series,
    settings: Settings,
    max_levels: int = 60,
) -> dict[str, Any]:
    """Build the payload for the category-mapping task.

    Category *levels* are sent verbatim even for masked columns would be useless here -
    the whole question is which spellings mean the same thing - so this function refuses
    to run on columns that must be masked. Callers check :func:`should_mask` first.
    """
    present = to_clean_strings(series)[~missing_mask(series)]
    counts = present.str.strip().value_counts().head(max_levels)
    return {
        "column": column.name,
        "distinct_values": int(column.unique_count),
        "levels": [{"value": str(value), "count": int(count)} for value, count in counts.items()],
    }


def findings_payload(findings: list[Finding], limit: int = 10) -> list[dict[str, Any]]:
    """Describe findings for the explanation task, without their example values."""
    return [
        {
            "finding_reference": finding.finding_id,
            "issue_type": finding.issue_type.value,
            "severity": finding.severity.value,
            "column": finding.column or "(whole dataset)",
            "affected_rows": finding.affected_rows,
            "total_rows": finding.row_count,
            "deterministic_explanation": finding.explanation,
        }
        for finding in findings[:limit]
    ]
