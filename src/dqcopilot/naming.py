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


#: Tokens suggesting a column holds an administrative or classification code. These are
#: stored as numbers and behave like numbers in every arithmetic sense, which is exactly
#: the trap: the median postcode of a city is a postcode, and it is meaningless. Only
#: the name distinguishes "municipality 3" from "quantity 3", so only the name is asked.
CODE_NAME_HINTS: tuple[str, ...] = (
    "zip",
    "zipcode",
    "postcode",
    "postal",
    "cap",
    "plz",
    "municipality",
    "municipio",
    "comune",
    "district",
    "region",
    "regione",
    "province",
    "provincia",
    "county",
    "canton",
    "prefecture",
    "borough",
    "ward",
    "zone",
    "sector",
    "nuts",
    "insee",
    "istat",
    "iso",
    "fips",
)


#: Tokens suggesting a column holds a position on the earth. Coordinates are genuine
#: measurements, so averaging them is arithmetically sound and factually useless: the
#: median longitude of a list of shops is a real address, and it is not this shop's.
COORDINATE_NAME_HINTS: tuple[str, ...] = (
    "lat",
    "latitude",
    "lon",
    "lng",
    "long",
    "longitude",
    "geo",
    "coord",
    "coordinate",
    "coordinates",
    "easting",
    "northing",
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


def looks_like_code(column: str) -> bool:
    """True when the column holds a label that happens to be written with digits.

    Identifiers and administrative codes are both included: neither can be averaged,
    summed or interpolated, so neither may be imputed from the other values in its
    column. The test is deliberately name-only. No property of the numbers themselves
    separates a district code from a count of items, and inventing a structural rule
    that appears to would only hide the guess.
    """
    return looks_like_identifier(column) or name_matches(column, CODE_NAME_HINTS)


def looks_like_coordinate(column: str) -> bool:
    """True when the column name suggests a geographic coordinate."""
    return name_matches(column, COORDINATE_NAME_HINTS)
