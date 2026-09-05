"""Pydantic models describing a dataset profile."""

from __future__ import annotations

from datetime import UTC, datetime

from pydantic import BaseModel, Field

from dqcopilot.models.enums import SemanticType


class ValueCount(BaseModel):
    """A value and how often it occurs in a column."""

    value: str
    count: int


class NumericStats(BaseModel):
    """Descriptive statistics for numeric-looking columns."""

    minimum: float | None = None
    maximum: float | None = None
    mean: float | None = None
    median: float | None = None
    std_dev: float | None = None
    zero_count: int = 0
    negative_count: int = 0


class TextStats(BaseModel):
    """Descriptive statistics for text columns."""

    min_length: int | None = None
    max_length: int | None = None
    mean_length: float | None = None
    whitespace_padded_count: int = 0
    empty_string_count: int = 0


class TemporalStats(BaseModel):
    """Descriptive statistics for date-like columns."""

    minimum: str | None = None
    maximum: str | None = None
    detected_formats: list[str] = Field(default_factory=list)
    unparseable_count: int = 0


class ColumnProfile(BaseModel):
    """Per-column profiling result."""

    name: str
    position: int
    pandas_dtype: str
    semantic_type: SemanticType
    semantic_type_confidence: float = Field(ge=0.0, le=1.0, default=0.0)

    row_count: int
    missing_count: int
    unique_count: int

    numeric: NumericStats | None = None
    text: TextStats | None = None
    temporal: TemporalStats | None = None

    top_values: list[ValueCount] = Field(default_factory=list)
    sample_values: list[str] = Field(default_factory=list)

    @property
    def missing_ratio(self) -> float:
        """Fraction of missing values in the column (0.0 - 1.0)."""
        return self.missing_count / self.row_count if self.row_count else 0.0

    @property
    def unique_ratio(self) -> float:
        """Fraction of distinct non-null values relative to row count."""
        return self.unique_count / self.row_count if self.row_count else 0.0

    @property
    def is_constant(self) -> bool:
        """True when the column holds at most one distinct non-null value."""
        return self.unique_count <= 1 and self.row_count > 1


class DatasetProfile(BaseModel):
    """Whole-dataset profiling result."""

    source_name: str
    row_count: int
    column_count: int
    duplicate_row_count: int = 0
    memory_bytes: int = 0
    profiled_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    columns: list[ColumnProfile] = Field(default_factory=list)

    def column(self, name: str) -> ColumnProfile | None:
        """Return the profile of ``name`` or ``None`` when the column is absent."""
        for col in self.columns:
            if col.name == name:
                return col
        return None

    @property
    def total_cells(self) -> int:
        """Number of cells in the dataset."""
        return self.row_count * self.column_count

    @property
    def total_missing(self) -> int:
        """Number of missing cells across all columns."""
        return sum(col.missing_count for col in self.columns)

    @property
    def missing_ratio(self) -> float:
        """Fraction of missing cells across the whole dataset."""
        return self.total_missing / self.total_cells if self.total_cells else 0.0
