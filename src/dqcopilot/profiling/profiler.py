"""Dataset profiling: per-column statistics and inferred semantic types."""

from __future__ import annotations

import time

import pandas as pd

from dqcopilot.logging_conf import get_logger
from dqcopilot.models.enums import SemanticType
from dqcopilot.models.profile import (
    ColumnProfile,
    DatasetProfile,
    NumericStats,
    TemporalStats,
    TextStats,
    ValueCount,
)
from dqcopilot.profiling.type_inference import (
    analyze_dates,
    coerce_numeric,
    format_label,
    infer_semantic_type,
    missing_mask,
    non_null_strings,
    to_clean_strings,
)

logger = get_logger(__name__)

TOP_VALUES_LIMIT = 10
SAMPLE_VALUES_LIMIT = 5


def profile_dataset(frame: pd.DataFrame, source_name: str = "dataset") -> DatasetProfile:
    """Build a full profile of ``frame``.

    Args:
        frame: The parsed dataset.
        source_name: Name shown in the report (usually the sanitised filename).

    Returns:
        A :class:`~dqcopilot.models.profile.DatasetProfile`.
    """
    started = time.perf_counter()
    row_count = int(frame.shape[0])

    columns = [
        profile_column(frame[name], name=str(name), position=position, row_count=row_count)
        for position, name in enumerate(frame.columns)
    ]

    profile = DatasetProfile(
        source_name=source_name,
        row_count=row_count,
        column_count=int(frame.shape[1]),
        duplicate_row_count=int(frame.duplicated(keep="first").sum()),
        memory_bytes=int(frame.memory_usage(deep=True).sum()),
        columns=columns,
    )

    logger.info(
        "Profiled dataset",
        extra={
            "rows": profile.row_count,
            "columns": profile.column_count,
            "duration_seconds": round(time.perf_counter() - started, 3),
        },
    )
    return profile


def profile_column(
    series: pd.Series,
    name: str,
    position: int,
    row_count: int,
) -> ColumnProfile:
    """Profile a single column."""
    semantic_type, confidence = infer_semantic_type(series, name)
    missing = missing_mask(series)
    present = series[~missing]

    profile = ColumnProfile(
        name=name,
        position=position,
        pandas_dtype=str(series.dtype),
        semantic_type=semantic_type,
        semantic_type_confidence=round(float(confidence), 4),
        row_count=row_count,
        missing_count=int(missing.sum()),
        unique_count=int(_normalised_values(present).nunique()),
        top_values=_top_values(present),
        sample_values=_sample_values(present),
    )

    if semantic_type.is_numeric or _has_numeric_evidence(present, semantic_type):
        profile.numeric = _numeric_stats(present)
    if semantic_type.is_temporal:
        profile.temporal = _temporal_stats(series)
    if not semantic_type.is_numeric and not semantic_type.is_temporal:
        profile.text = _text_stats(series)

    return profile


# --------------------------------------------------------------------------- helpers


def _normalised_values(series: pd.Series) -> pd.Series:
    """Return values as trimmed strings for cardinality counting."""
    if series.dtype == object or pd.api.types.is_string_dtype(series):
        return to_clean_strings(series).str.strip()
    return series


def _has_numeric_evidence(series: pd.Series, semantic_type: SemanticType) -> bool:
    """True when a non-numeric column still holds mostly numeric-looking values."""
    if semantic_type in (SemanticType.EMPTY, SemanticType.DATE, SemanticType.DATETIME):
        return False
    if series.empty:
        return False
    return float(coerce_numeric(series).notna().mean()) >= 0.5


def _numeric_stats(series: pd.Series) -> NumericStats:
    numeric = coerce_numeric(series).dropna()
    if numeric.empty:
        return NumericStats()
    return NumericStats(
        minimum=float(numeric.min()),
        maximum=float(numeric.max()),
        mean=float(numeric.mean()),
        median=float(numeric.median()),
        std_dev=float(numeric.std()) if numeric.size > 1 else 0.0,
        zero_count=int((numeric == 0).sum()),
        negative_count=int((numeric < 0).sum()),
    )


def _text_stats(series: pd.Series) -> TextStats:
    strings = non_null_strings(series)
    if strings.empty:
        return TextStats()
    lengths = strings.str.len()
    stripped = strings.str.strip()
    return TextStats(
        min_length=int(lengths.min()),
        max_length=int(lengths.max()),
        mean_length=round(float(lengths.mean()), 2),
        whitespace_padded_count=int((stripped != strings).sum()),
        empty_string_count=int((stripped == "").sum()),
    )


def _temporal_stats(series: pd.Series) -> TemporalStats:
    analysis = analyze_dates(series)
    parsed = pd.to_datetime(
        to_clean_strings(series) if series.dtype == object else series,
        errors="coerce",
        format="mixed",
        dayfirst=True,
    )
    valid = parsed.dropna()
    return TemporalStats(
        minimum=str(valid.min().date()) if not valid.empty else None,
        maximum=str(valid.max().date()) if not valid.empty else None,
        detected_formats=sorted(format_label(fmt) for fmt in analysis.format_counts),
        unparseable_count=analysis.total_values - analysis.parsed_count,
    )


def _top_values(series: pd.Series, limit: int = TOP_VALUES_LIMIT) -> list[ValueCount]:
    if series.empty:
        return []
    counts = _normalised_values(series).value_counts().head(limit)
    return [ValueCount(value=str(value), count=int(count)) for value, count in counts.items()]


def _sample_values(series: pd.Series, limit: int = SAMPLE_VALUES_LIMIT) -> list[str]:
    if series.empty:
        return []
    unique = pd.Series(series.unique()).dropna().head(limit)
    return [str(value) for value in unique]
