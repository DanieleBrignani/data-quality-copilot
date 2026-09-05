"""Export cleaned datasets as CSV or XLSX.

Security note: a cell beginning with ``=``, ``+``, ``-`` or ``@`` is interpreted as a
formula by Excel and LibreOffice when a CSV is opened. Since this application hands the
user a file built from data someone else uploaded, every export neutralises those cells
by prefixing a single quote. That is the standard mitigation for CSV injection.
"""

from __future__ import annotations

import io
import re

import pandas as pd

from dqcopilot.ingestion.sanitize import sanitize_filename

#: Leading characters that spreadsheet software treats as the start of a formula.
FORMULA_PREFIXES = ("=", "+", "-", "@", "\t", "\r")

_SAFE_NUMBER = re.compile(r"^[-+]?\d+(?:[.,]\d+)?(?:[eE][-+]?\d+)?$")


def neutralise_formula(value: object) -> object:
    """Prefix a formula-looking string with ``'`` so spreadsheets treat it as text.

    Plain negative numbers such as ``-42.5`` are left alone: they start with ``-`` but
    are data, not formulas.
    """
    if not isinstance(value, str):
        return value
    text = value.lstrip()
    if not text or not text.startswith(FORMULA_PREFIXES):
        return value
    if _SAFE_NUMBER.match(text):
        return value
    return f"'{value}"


def sanitise_for_export(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``frame`` with every formula-looking cell neutralised."""
    safe = frame.copy()
    for column in safe.columns:
        series = safe[column]
        if series.dtype == object or pd.api.types.is_string_dtype(series):
            safe[column] = series.map(neutralise_formula, na_action="ignore")
    return safe


def to_csv_bytes(frame: pd.DataFrame, sanitise: bool = True) -> bytes:
    """Serialise ``frame`` as UTF-8 CSV bytes (with a BOM, so Excel opens it correctly)."""
    payload = sanitise_for_export(frame) if sanitise else frame
    buffer = io.StringIO()
    payload.to_csv(buffer, index=False, lineterminator="\n")
    return buffer.getvalue().encode("utf-8-sig")


def to_xlsx_bytes(frame: pd.DataFrame, sheet_name: str = "cleaned", sanitise: bool = True) -> bytes:
    """Serialise ``frame`` as an XLSX workbook."""
    payload = sanitise_for_export(frame) if sanitise else frame
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        payload.to_excel(writer, index=False, sheet_name=sheet_name[:31])
    return buffer.getvalue()


def cleaned_filename(source_name: str, extension: str = ".csv") -> str:
    """Build the download name for a cleaned dataset."""
    stem = sanitize_filename(source_name).rsplit(".", 1)[0] or "dataset"
    return f"{stem}_cleaned{extension}"
