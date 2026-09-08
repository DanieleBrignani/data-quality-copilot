"""Enumerations shared across the domain model."""

from __future__ import annotations

from enum import StrEnum


class Severity(StrEnum):
    """Severity of a data quality finding, ordered from worst to mildest."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"

    @property
    def weight(self) -> float:
        """Penalty weight used by the quality score (documented in the README)."""
        return _SEVERITY_WEIGHTS[self]

    @property
    def rank(self) -> int:
        """Sort rank; lower means more severe."""
        return _SEVERITY_RANK[self]


_SEVERITY_WEIGHTS: dict[Severity, float] = {
    Severity.CRITICAL: 1.0,
    Severity.HIGH: 0.6,
    Severity.MEDIUM: 0.3,
    Severity.LOW: 0.1,
    Severity.INFO: 0.0,
}

_SEVERITY_RANK: dict[Severity, int] = {
    Severity.CRITICAL: 0,
    Severity.HIGH: 1,
    Severity.MEDIUM: 2,
    Severity.LOW: 3,
    Severity.INFO: 4,
}


class IssueType(StrEnum):
    """Catalogue of detectable data quality problems."""

    MISSING_VALUES = "missing_values"
    EXACT_DUPLICATE_ROWS = "exact_duplicate_rows"
    PROBABLE_DUPLICATE_ROWS = "probable_duplicate_rows"
    INVALID_DATE = "invalid_date"
    INCONSISTENT_DATE_FORMAT = "inconsistent_date_format"
    FUTURE_DATE = "future_date"
    INVALID_EMAIL = "invalid_email"
    NUMERIC_STORED_AS_TEXT = "numeric_stored_as_text"
    IMPOSSIBLE_NUMERIC = "impossible_numeric"
    SUSPICIOUS_NUMERIC = "suspicious_numeric"
    INCONSISTENT_CATEGORY = "inconsistent_category"
    LEADING_TRAILING_WHITESPACE = "leading_trailing_whitespace"
    INCONSISTENT_CAPITALIZATION = "inconsistent_capitalization"
    MIXED_DATATYPES = "mixed_datatypes"
    SCHEMA_MISMATCH = "schema_mismatch"
    BUSINESS_RULE_VIOLATION = "business_rule_violation"
    CONSTANT_COLUMN = "constant_column"
    CORRUPTED_ENCODING = "corrupted_encoding"
    PLACEHOLDER_VALUE = "placeholder_value"


class SemanticType(StrEnum):
    """Inferred semantic type of a column."""

    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    DATE = "date"
    DATETIME = "datetime"
    EMAIL = "email"
    CATEGORICAL = "categorical"
    TEXT = "text"
    IDENTIFIER = "identifier"
    EMPTY = "empty"
    UNKNOWN = "unknown"

    @property
    def is_numeric(self) -> bool:
        """True for numeric semantic types."""
        return self in (SemanticType.INTEGER, SemanticType.FLOAT)

    @property
    def is_temporal(self) -> bool:
        """True for date-like semantic types."""
        return self in (SemanticType.DATE, SemanticType.DATETIME)


class FindingSource(StrEnum):
    """Where a finding came from.

    Deterministic findings are produced by Python code and are reproducible.
    AI findings are suggestions from the Anthropic API and are always advisory.
    """

    DETERMINISTIC = "deterministic"
    AI = "ai"


class CorrectionAction(StrEnum):
    """Machine-applicable correction operations."""

    STRIP_WHITESPACE = "strip_whitespace"
    NORMALIZE_CASE = "normalize_case"
    FILL_MISSING = "fill_missing"
    FILL_FROM_RELATED = "fill_from_related"
    DROP_DUPLICATE_ROWS = "drop_duplicate_rows"
    CAST_TO_NUMERIC = "cast_to_numeric"
    PARSE_DATES = "parse_dates"
    MAP_CATEGORY = "map_category"
    REPAIR_ENCODING = "repair_encoding"
    CLEAR_INVALID_VALUES = "clear_invalid_values"
    CLIP_TO_RANGE = "clip_to_range"
    MANUAL_REVIEW = "manual_review"

    @property
    def is_applicable(self) -> bool:
        """False for actions that only flag a problem for a human."""
        return self is not CorrectionAction.MANUAL_REVIEW


class DecisionStatus(StrEnum):
    """Approval state of a proposed correction."""

    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
