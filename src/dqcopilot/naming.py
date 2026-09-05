"""Column-name heuristics shared by profiling and validation.

Column names are weak evidence, but they are the only evidence available for questions
like "may this date be in the future?". Matching is done on whole word tokens rather
than substrings, because substring matching produces confident nonsense: ``"discount"``
contains ``"count"`` and ``"average"`` contains ``"age"``.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

_TOKEN_SPLIT = re.compile(r"[^a-z0-9]+")

#: Tokens suggesting a column holds a date or timestamp.
TEMPORAL_NAME_HINTS: tuple[str, ...] = (
    "date",
    "dates",
    "day",
    "time",
    "timestamp",
    "datetime",
    "dob",
    "birthday",
    "on",
    "at",
    "since",
    "until",
)

#: Tokens suggesting a column holds an identifier rather than a measurement.
IDENTIFIER_NAME_HINTS: tuple[str, ...] = (
    "id",
    "ids",
    "code",
    "reference",
    "ref",
    "sku",
    "vat",
    "uuid",
    "guid",
    "number",
    "no",
    "siren",
    "siret",
    "ean",
    "isbn",
)


def column_tokens(name: str) -> set[str]:
    """Split a column name into lowercase word tokens.

    ``"total_amount_2023"`` becomes ``{"total", "amount", "2023"}``.
    """
    return {token for token in _TOKEN_SPLIT.split(name.lower()) if token}


def name_matches(column: str, hints: Sequence[str]) -> bool:
    """True when a column name contains one of ``hints`` as a whole word."""
    tokens = column_tokens(column)
    return any(hint in tokens for hint in hints)


def looks_temporal(column: str) -> bool:
    """True when the column name suggests a date or timestamp."""
    return name_matches(column, TEMPORAL_NAME_HINTS)


def looks_like_identifier(column: str) -> bool:
    """True when the column name suggests an identifier."""
    return name_matches(column, IDENTIFIER_NAME_HINTS)
