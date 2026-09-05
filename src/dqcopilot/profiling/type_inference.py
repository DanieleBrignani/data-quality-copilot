"""Deterministic type inference and value-coercion helpers.

Everything here is pure Python/pandas and fully reproducible. The rest of the
application (profiling, deterministic checks, correction previews) shares these
primitives so that "what the profiler saw" and "what the fix would do" can never
drift apart.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from functools import lru_cache

import numpy as np
import pandas as pd

from dqcopilot.models.enums import SemanticType
from dqcopilot.naming import looks_like_identifier, looks_temporal

# --------------------------------------------------------------------------- regexes

#: Pragmatic email pattern. Deliberately stricter than RFC 5322 (which allows
#: addresses nobody actually uses) and looser than a deliverability check.
EMAIL_RE = re.compile(
    r"^[A-Za-z0-9!#$%&'*+/=?^_`{|}~-]+"
    r"(?:\.[A-Za-z0-9!#$%&'*+/=?^_`{|}~-]+)*"
    r"@[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?"
    r"(?:\.[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?)+$"
)

_BOOLEAN_TRUE = frozenset({"true", "yes", "y", "1", "t", "si", "sì", "oui", "vrai"})
_BOOLEAN_FALSE = frozenset({"false", "no", "n", "0", "f", "non", "faux"})

_CURRENCY_CHARS = "€$£¥₹"
_WHITESPACE_CHARS = " \t   "
_NUMERIC_STRIP = re.compile(rf"[{_CURRENCY_CHARS}{_WHITESPACE_CHARS}]")
_TRAILING_PERCENT = re.compile(r"%$")
_PARENTHESISED = re.compile(r"^\((.*)\)$")
_DIGITS_ONLY = re.compile(r"^[+-]?\d+$")

#: Date formats probed in order. Each entry is (strptime format, human label).
DATE_FORMATS: tuple[tuple[str, str], ...] = (
    ("%Y-%m-%d", "YYYY-MM-DD (ISO)"),
    ("%d/%m/%Y", "DD/MM/YYYY"),
    ("%m/%d/%Y", "MM/DD/YYYY"),
    ("%d-%m-%Y", "DD-MM-YYYY"),
    ("%Y/%m/%d", "YYYY/MM/DD"),
    ("%d.%m.%Y", "DD.MM.YYYY"),
    ("%d %b %Y", "DD Mon YYYY"),
    ("%b %d, %Y", "Mon DD, YYYY"),
    ("%d %B %Y", "DD Month YYYY"),
    ("%B %d, %Y", "Month DD, YYYY"),
    ("%Y%m%d", "YYYYMMDD"),
    ("%d/%m/%y", "DD/MM/YY"),
    ("%m/%d/%y", "MM/DD/YY"),
    ("%Y-%m-%d %H:%M:%S", "YYYY-MM-DD HH:MM:SS"),
    ("%Y-%m-%dT%H:%M:%S", "ISO 8601 datetime"),
    ("%d/%m/%Y %H:%M", "DD/MM/YYYY HH:MM"),
    ("%m/%d/%Y %H:%M", "MM/DD/YYYY HH:MM"),
)

_FORMAT_LABELS: dict[str, str] = dict(DATE_FORMATS)

#: A string this long or longer is treated as free text rather than a category.
MAX_CATEGORY_LENGTH = 60
#: Column is considered categorical below this ratio of distinct values.
CATEGORICAL_UNIQUE_RATIO = 0.25
#: ...or below this absolute number of distinct values.
CATEGORICAL_MAX_LEVELS = 50
#: Type inference reads at most this many values per column (evenly spaced, so the
#: result is deterministic and unaffected by how the file happens to be sorted).
INFERENCE_SAMPLE_SIZE = 2000
#: Cap on the number of unparseable row indices retained by :func:`analyze_dates`.
MAX_UNPARSEABLE_SAMPLE = 200
#: Share of values that must parse as dates before a column is called temporal.
DATE_RATIO_UNNAMED = 0.7
#: Lower bar used when the column *name* already says the column holds dates.
DATE_RATIO_NAMED = 0.4


# --------------------------------------------------------------------------- helpers


def to_clean_strings(series: pd.Series) -> pd.Series:
    """Return ``series`` as strings with ``NaN`` preserved.

    Values are **not** trimmed: preserving the original spacing is what allows the
    whitespace check to work.
    """
    if series.dtype == object or pd.api.types.is_string_dtype(series):
        return series.astype("string")
    return series.astype("string")


def non_null_strings(series: pd.Series) -> pd.Series:
    """Return the non-null values of ``series`` as a string Series."""
    return to_clean_strings(series).dropna()


def is_blank_string(value: object) -> bool:
    """True when ``value`` is a string made only of whitespace."""
    return isinstance(value, str) and not value.strip()


def missing_mask(series: pd.Series) -> pd.Series:
    """Return a boolean mask of missing values.

    A value counts as missing when it is ``NaN``/``None`` **or** a string containing
    only whitespace. Using one definition everywhere keeps the profile, the checks and
    the corrections consistent.
    """
    mask = series.isna()
    if series.dtype == object or pd.api.types.is_string_dtype(series):
        blank = to_clean_strings(series).str.strip().eq("").fillna(False)
        mask = mask | blank.astype(bool)
    return mask.astype(bool)


def evenly_spaced_sample(series: pd.Series, limit: int) -> pd.Series:
    """Return at most ``limit`` values taken at a constant stride.

    Sampling with a stride (rather than the first N rows or a random draw) keeps type
    inference both deterministic and robust to files that are sorted by one column.
    """
    size = int(series.size)
    if size <= limit or limit <= 0:
        return series
    stride = -(-size // limit)  # ceiling division
    return series.iloc[::stride]


# --------------------------------------------------------------------------- numeric


def parse_numeric_token(token: str) -> float | None:
    """Parse a single human-written numeric token.

    Handles currency symbols, thin/non-breaking spaces, thousands separators in both
    the ``1,234.56`` and ``1.234,56`` conventions, accounting negatives ``(500)`` and a
    trailing percent sign (the sign is dropped, the magnitude is kept).

    Args:
        token: The raw string to parse.

    Returns:
        The parsed float, or ``None`` when the token is not numeric.
    """
    text = token.strip()
    if not text:
        return None

    negative = False
    match = _PARENTHESISED.match(text)
    if match:
        negative = True
        text = match.group(1).strip()

    text = _TRAILING_PERCENT.sub("", text).strip()
    text = _NUMERIC_STRIP.sub("", text)
    if not text:
        return None

    if text[0] in "+-":
        negative = negative or text[0] == "-"
        text = text[1:]
    if not text:
        return None

    normalised = _normalise_separators(text)
    if normalised is None:
        return None

    try:
        value = float(normalised)
    except ValueError:
        return None
    return -value if negative else value


def _normalise_separators(text: str) -> str | None:
    """Convert a digit group with ``.``/``,`` separators into a plain float string."""
    has_dot = "." in text
    has_comma = "," in text

    if has_dot and has_comma:
        decimal_sep = "." if text.rfind(".") > text.rfind(",") else ","
        thousands_sep = "," if decimal_sep == "." else "."
        text = text.replace(thousands_sep, "")
        text = text.replace(decimal_sep, ".")
    elif has_comma:
        text = _resolve_single_separator(text, ",")
    elif has_dot:
        text = _resolve_single_separator(text, ".")

    return text if re.fullmatch(r"\d*\.?\d+(?:[eE][+-]?\d+)?", text) else None


def _resolve_single_separator(text: str, separator: str) -> str:
    """Decide whether a lone ``.``/``,`` is a decimal point or a thousands separator."""
    parts = text.split(separator)
    if len(parts) > 2:
        # 1.234.567 -> repeated separators can only be thousands groupings.
        return "".join(parts)
    tail = parts[1]
    if len(tail) == 3 and len(parts[0]) <= 3 and parts[0] != "":
        # Ambiguous (e.g. "1,234"): the thousands reading is far more common.
        return "".join(parts)
    return ".".join(parts)


def coerce_numeric(series: pd.Series) -> pd.Series:
    """Coerce a Series to float using :func:`parse_numeric_token`.

    Args:
        series: Any Series (numeric passes through unchanged).

    Returns:
        A float Series with ``NaN`` wherever the value could not be parsed.
    """
    if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
        return series.astype("float64")

    strings = to_clean_strings(series)
    parsed = strings.map(
        lambda value: parse_numeric_token(value) if isinstance(value, str) else None,
        na_action="ignore",
    )
    return pd.to_numeric(parsed, errors="coerce").astype("float64")


def looks_like_integer(values: pd.Series) -> bool:
    """True when every parsed numeric value is a whole number."""
    numeric = values.dropna()
    if numeric.empty:
        return False
    return bool(np.all(np.isfinite(numeric)) and np.all(numeric == numeric.round()))


# --------------------------------------------------------------------------- dates


@lru_cache(maxsize=16384)
def match_date_formats(token: str) -> frozenset[str]:
    """Return every format from :data:`DATE_FORMATS` that parses ``token``.

    Cached: real datasets repeat values heavily, and probing 17 formats per distinct
    value is the single most expensive operation in the profiler.
    """
    text = token.strip()
    if not text:
        return frozenset()
    if _DIGITS_ONLY.match(text) and len(text) != 8:
        # Bare integers are not dates (a stray "2024" is a year, not a date).
        return frozenset()

    matches = {fmt for fmt, _ in DATE_FORMATS if _try_strptime(text, fmt) is not None}
    return frozenset(matches)


def _try_strptime(text: str, fmt: str) -> datetime | None:
    try:
        return datetime.strptime(text, fmt)  # noqa: DTZ007 - naive dates are intended
    except ValueError:
        return None


def format_label(fmt: str) -> str:
    """Return a human readable label for a strptime format."""
    return _FORMAT_LABELS.get(fmt, fmt)


@dataclass(slots=True)
class DateColumnAnalysis:
    """Result of analysing a column as dates."""

    total_values: int = 0
    parsed_count: int = 0
    unparseable_indices: list[int] = field(default_factory=list)
    #: Formats that explain *every* parseable value in the column.
    consistent_formats: frozenset[str] = frozenset()
    #: All formats observed, mapped to how many values they explain.
    format_counts: dict[str, int] = field(default_factory=dict)
    parsed: pd.Series | None = None

    @property
    def parse_ratio(self) -> float:
        """Fraction of non-null values that parsed as a date."""
        return self.parsed_count / self.total_values if self.total_values else 0.0

    @property
    def is_format_consistent(self) -> bool:
        """True when a single format explains all parseable values."""
        return bool(self.consistent_formats)


def analyze_dates(series: pd.Series) -> DateColumnAnalysis:
    """Analyse a Series for date parseability and format consistency.

    A column is *format consistent* when at least one format in
    :data:`DATE_FORMATS` explains every non-null value. Ambiguous values such as
    ``03/04/2024`` legitimately match several formats; consistency is therefore
    decided on the intersection of the per-value format sets, not on a single guess.

    Args:
        series: The column to analyse.

    Returns:
        A :class:`DateColumnAnalysis`.
    """
    analysis = DateColumnAnalysis()

    if pd.api.types.is_datetime64_any_dtype(series):
        values = series.dropna()
        analysis.total_values = int(values.size)
        analysis.parsed_count = int(values.size)
        analysis.consistent_formats = frozenset({"native"})
        analysis.format_counts = {"native": int(values.size)}
        analysis.parsed = pd.to_datetime(series, errors="coerce")
        return analysis

    strings = non_null_strings(series)
    analysis.total_values = int(strings.size)
    if analysis.total_values == 0:
        return analysis

    running: frozenset[str] | None = None
    counts: dict[str, int] = {}

    labels = strings.index.to_numpy()
    for position, value in enumerate(strings.to_numpy()):
        matches = match_date_formats(str(value))
        if not matches:
            if len(analysis.unparseable_indices) < MAX_UNPARSEABLE_SAMPLE:
                analysis.unparseable_indices.append(int(labels[position]))
            continue
        analysis.parsed_count += 1
        for fmt in matches:
            counts[fmt] = counts.get(fmt, 0) + 1
        running = matches if running is None else (running & matches)

    analysis.consistent_formats = running or frozenset()
    analysis.format_counts = counts
    return analysis


def parse_dates_with_format(series: pd.Series, fmt: str | None) -> pd.Series:
    """Parse ``series`` into datetimes, using ``fmt`` when provided."""
    if pd.api.types.is_datetime64_any_dtype(series):
        return pd.to_datetime(series, errors="coerce")
    strings = to_clean_strings(series)
    if fmt and fmt != "native":
        return pd.to_datetime(strings, format=fmt, errors="coerce")
    return pd.to_datetime(strings, errors="coerce", format="mixed", dayfirst=True)


# --------------------------------------------------------------------------- booleans


def parse_boolean_token(token: str) -> bool | None:
    """Parse a human-written boolean token, or return ``None``."""
    text = token.strip().lower()
    if text in _BOOLEAN_TRUE:
        return True
    if text in _BOOLEAN_FALSE:
        return False
    return None


# --------------------------------------------------------------------------- inference


def infer_semantic_type(series: pd.Series, column_name: str = "") -> tuple[SemanticType, float]:
    """Infer the semantic type of a column.

    The rules are intentionally simple and ordered from most to least specific so the
    outcome is easy to explain to a reviewer:

    1. native pandas dtypes (Excel already knows its types);
    2. empty column;
    3. email, when most values match the email pattern;
    4. boolean, date/datetime, numeric, when most values parse;
    5. identifier, when values are unique and short;
    6. categorical, when the number of distinct values is small;
    7. text otherwise.

    Args:
        series: The column to inspect.
        column_name: Used only as a weak hint for identifier detection.

    Returns:
        A tuple of the inferred type and a confidence between 0 and 1.
    """
    if pd.api.types.is_bool_dtype(series):
        return SemanticType.BOOLEAN, 1.0
    if pd.api.types.is_datetime64_any_dtype(series):
        return SemanticType.DATETIME, 1.0
    if pd.api.types.is_integer_dtype(series):
        return SemanticType.INTEGER, 1.0
    if pd.api.types.is_float_dtype(series):
        return SemanticType.FLOAT, 1.0

    values = non_null_strings(series)
    if values.empty:
        return SemanticType.EMPTY, 1.0

    values = evenly_spaced_sample(values, INFERENCE_SAMPLE_SIZE)
    total = int(values.size)
    stripped = values.str.strip()

    email_ratio = float(stripped.map(lambda v: bool(EMAIL_RE.match(v))).mean())
    at_ratio = float(stripped.str.contains("@", regex=False).mean())
    if email_ratio >= 0.6 or (at_ratio >= 0.8 and email_ratio >= 0.3):
        return SemanticType.EMAIL, max(email_ratio, 0.6)

    boolean_ratio = float(stripped.map(lambda v: parse_boolean_token(v) is not None).mean())
    if boolean_ratio >= 0.9:
        return SemanticType.BOOLEAN, boolean_ratio

    # A column named "signup_date" whose values half fail to parse is still a date
    # column with bad data in it - and treating it as one is what lets the invalid-date
    # check explain the problem. The name lowers the bar; it never raises it.
    date_threshold = DATE_RATIO_NAMED if looks_temporal(column_name) else DATE_RATIO_UNNAMED
    date_analysis = analyze_dates(values)
    date_ratio = date_analysis.parse_ratio
    if date_ratio >= date_threshold:
        has_time = any("%H" in fmt for fmt in date_analysis.format_counts)
        return SemanticType.DATETIME if has_time else SemanticType.DATE, date_ratio

    numeric = coerce_numeric(values)
    numeric_ratio = float(numeric.notna().sum()) / total
    if numeric_ratio >= 0.7:
        kind = SemanticType.INTEGER if looks_like_integer(numeric) else SemanticType.FLOAT
        return kind, numeric_ratio

    distinct = int(stripped.nunique())
    max_length = int(stripped.str.len().max())
    unique_ratio = distinct / total

    # Identifier detection is deliberately conservative. "Every value is distinct" is
    # weak evidence in a small sample - eight unrelated words are all distinct too - so
    # a column also needs a name that says "identifier", or enough rows for uniqueness
    # to mean something. Columns that are mostly numbers or dates are never identifiers.
    if (
        unique_ratio > 0.95
        and max_length <= 40
        and numeric_ratio < 0.5
        and date_ratio < 0.5
        and (looks_like_identifier(column_name) or (unique_ratio == 1.0 and total >= 20))
    ):
        return SemanticType.IDENTIFIER, unique_ratio

    if max_length <= MAX_CATEGORY_LENGTH and (
        distinct <= CATEGORICAL_MAX_LEVELS or unique_ratio <= CATEGORICAL_UNIQUE_RATIO
    ):
        return SemanticType.CATEGORICAL, 1.0 - unique_ratio

    return SemanticType.TEXT, 0.5
