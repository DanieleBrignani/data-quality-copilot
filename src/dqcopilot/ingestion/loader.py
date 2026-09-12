"""Load CSV and XLSX uploads into a pandas DataFrame.

Design decision: CSV files are read as **raw strings** (``dtype=str``) instead of letting
pandas guess types. A data quality tool must be able to see the original text in order to
detect numbers stored as text, mixed date formats, stray whitespace and so on. Excel files
keep their native cell types, because Excel already carries real type information that is
itself useful evidence.
"""

from __future__ import annotations

import csv
import io
import re
import time
from dataclasses import dataclass, field

import pandas as pd

from dqcopilot.config import Settings, get_settings
from dqcopilot.ingestion.errors import (
    CorruptFileError,
    DatasetTooLargeError,
    EmptyFileError,
    IngestionError,
)
from dqcopilot.ingestion.sanitize import (
    sanitize_filename,
    validate_content_matches_extension,
    validate_extension,
)
from dqcopilot.logging_conf import get_logger

logger = get_logger(__name__)

#: The reader reads; only the checks judge.
#:
#: Pandas converts about twenty tokens to "missing" while parsing - ``NA``, ``null``,
#: ``None``, ``#N/A`` - and this project used to add ``unknown``, ``missing``, ``nil``,
#: ``--`` and ``?`` on top. Every one of those is a value somebody wrote in a cell, and
#: rewriting it during the read is a silent edit: not approved, absent from the audit
#: log, and gone from the exported file. It also hid the defect from the very check
#: written to surface it, since ``placeholder_values`` can only report what reaches it.
#:
#: So nothing is converted. A cell is missing when it is empty or holds only whitespace,
#: which is what :func:`~dqcopilot.profiling.type_inference.missing_mask` already means
#: everywhere else. A cell containing the word "unknown" is reported as a placeholder and
#: cleared only if a human approves it.
CONVERT_NOTHING_TO_MISSING = False

_ENCODING_CANDIDATES: tuple[str, ...] = ("utf-8-sig", "utf-8", "cp1252", "latin-1")
_DELIMITER_CANDIDATES: tuple[str, ...] = (",", ";", "\t", "|")
_SNIFF_BYTES = 64 * 1024
_SNIFF_LINES = 25

#: Placeholder pandas assigns to a column whose header cell is empty.
_UNNAMED_RE = re.compile(r"^Unnamed:\s*\d+(?:_level_\d+)?$")


@dataclass(slots=True)
class LoadedDataset:
    """A parsed dataset plus the metadata gathered while reading it."""

    frame: pd.DataFrame
    source_name: str
    extension: str
    size_bytes: int
    original_columns: list[str]
    encoding: str | None = None
    delimiter: str | None = None
    sheet_name: str | None = None
    load_seconds: float = 0.0
    notes: list[str] = field(default_factory=list)

    @property
    def row_count(self) -> int:
        """Number of data rows."""
        return int(self.frame.shape[0])

    @property
    def column_count(self) -> int:
        """Number of columns."""
        return int(self.frame.shape[1])


def load_tabular(
    content: bytes,
    filename: str,
    settings: Settings | None = None,
) -> LoadedDataset:
    """Validate and parse an uploaded CSV or XLSX file.

    Args:
        content: Raw bytes of the uploaded file.
        filename: Original filename supplied by the client.
        settings: Optional settings override (defaults to the cached application settings).

    Returns:
        A :class:`LoadedDataset`.

    Raises:
        FileTooLargeError: The upload exceeds ``MAX_UPLOAD_MB``.
        UnsupportedFileTypeError: Extension or content type is not supported.
        EmptyFileError: The file or the parsed table is empty.
        CorruptFileError: The file could not be parsed.
        DatasetTooLargeError: The table exceeds the configured row/column limits.
    """
    settings = settings or get_settings()
    started = time.perf_counter()

    safe_name = sanitize_filename(filename)
    _validate_size(content, settings)
    extension = validate_extension(safe_name)
    validate_content_matches_extension(content, extension)

    if extension == ".csv":
        dataset = _load_csv(content, safe_name, settings)
    else:
        dataset = _load_xlsx(content, safe_name, settings)

    dataset.size_bytes = len(content)
    dataset.load_seconds = time.perf_counter() - started

    _validate_shape(dataset, settings)
    _normalise_headers(dataset)

    logger.info(
        "Loaded dataset",
        extra={
            "rows": dataset.row_count,
            "columns": dataset.column_count,
            "size_bytes": dataset.size_bytes,
            "extension": dataset.extension,
            "load_seconds": round(dataset.load_seconds, 3),
        },
    )
    return dataset


# --------------------------------------------------------------------------- helpers


def _validate_size(content: bytes, settings: Settings) -> None:
    from dqcopilot.ingestion.errors import FileTooLargeError

    if not content:
        raise EmptyFileError("The uploaded file is empty (0 bytes).")
    if len(content) > settings.max_upload_bytes:
        actual_mb = len(content) / (1024 * 1024)
        raise FileTooLargeError(
            f"File is {actual_mb:.1f} MB but the limit is {settings.max_upload_mb:.0f} MB."
        )


def _decode(content: bytes) -> tuple[str, str]:
    """Decode ``content`` using the first encoding that succeeds."""
    for encoding in _ENCODING_CANDIDATES:
        try:
            return content.decode(encoding), encoding
        except UnicodeDecodeError:
            continue
    # latin-1 never fails, so reaching this point means something is very wrong.
    raise CorruptFileError("The CSV file could not be decoded as text.")


def _sniff_delimiter(sample: str) -> str:
    """Detect the delimiter of a CSV sample.

    Only characters that actually appear in the **header** line are considered, and the
    winner is the one that splits the first rows into the most consistent number of
    fields. Restricting candidates to the header matters: without it, a single-column
    file whose values contain commas (``"1,234.50"``) would be split into two columns.
    """
    lines = [line for line in sample.splitlines() if line.strip()][:_SNIFF_LINES]
    if not lines:
        return ","

    candidates = [candidate for candidate in _DELIMITER_CANDIDATES if candidate in lines[0]]
    if not candidates:
        return ","

    best = ","
    best_score = (-1.0, 0)
    for candidate in candidates:
        try:
            rows = [row for row in csv.reader(lines, delimiter=candidate) if row]
        except csv.Error:
            continue
        if not rows:
            continue
        expected = len(rows[0])
        if expected < 2:
            continue
        consistency = sum(1 for row in rows if len(row) == expected) / len(rows)
        score = (consistency, expected)
        if score > best_score:
            best_score, best = score, candidate
    return best


def _load_csv(content: bytes, safe_name: str, settings: Settings) -> LoadedDataset:
    text, encoding = _decode(content)
    if not text.strip():
        raise EmptyFileError("The CSV file contains no readable text.")

    delimiter = _sniff_delimiter(text[:_SNIFF_BYTES])
    try:
        frame = pd.read_csv(
            io.StringIO(text),
            sep=delimiter,
            dtype=str,
            keep_default_na=CONVERT_NOTHING_TO_MISSING,
            skipinitialspace=False,
            skip_blank_lines=True,
            nrows=settings.max_rows + 1,
        )
    except pd.errors.EmptyDataError as exc:
        raise EmptyFileError("The CSV file contains no columns.") from exc
    except (pd.errors.ParserError, ValueError) as exc:
        raise CorruptFileError(
            "The CSV file could not be parsed. Check for unbalanced quotes or "
            "rows with an inconsistent number of fields."
        ) from exc

    return LoadedDataset(
        frame=frame,
        source_name=safe_name,
        extension=".csv",
        size_bytes=len(content),
        original_columns=[str(col) for col in frame.columns],
        encoding=encoding,
        delimiter=delimiter,
    )


def _load_xlsx(content: bytes, safe_name: str, settings: Settings) -> LoadedDataset:
    """Read the first worksheet of an XLSX file.

    ``pandas`` opens workbooks through ``openpyxl`` with ``data_only=True``, so cached
    values are read and formulas are never evaluated. Macro-enabled formats (``.xlsm``)
    are rejected earlier by the extension check.
    """
    try:
        with pd.ExcelFile(io.BytesIO(content), engine="openpyxl") as workbook:
            if not workbook.sheet_names:
                raise EmptyFileError("The workbook contains no worksheets.")
            sheet_name = str(workbook.sheet_names[0])
            frame = workbook.parse(
                sheet_name=sheet_name,
                keep_default_na=CONVERT_NOTHING_TO_MISSING,
                nrows=settings.max_rows + 1,
            )
    except IngestionError:
        raise
    except Exception as exc:  # noqa: BLE001 - openpyxl raises a wide range of errors
        raise CorruptFileError(
            "The Excel workbook could not be read. It may be corrupted or password protected."
        ) from exc

    return LoadedDataset(
        frame=frame,
        source_name=safe_name,
        extension=".xlsx",
        size_bytes=len(content),
        original_columns=[str(col) for col in frame.columns],
        sheet_name=sheet_name,
    )


def _validate_shape(dataset: LoadedDataset, settings: Settings) -> None:
    rows, columns = dataset.frame.shape
    if columns == 0:
        raise EmptyFileError("The file contains no columns.")
    if rows == 0:
        raise EmptyFileError("The file contains a header but no data rows.")
    if rows > settings.max_rows:
        raise DatasetTooLargeError(
            f"The dataset has more than {settings.max_rows:,} rows, which is above the "
            "limit configured for this demo."
        )
    if columns > settings.max_columns:
        raise DatasetTooLargeError(
            f"The dataset has {columns:,} columns, above the limit of {settings.max_columns:,}."
        )


def _normalise_headers(dataset: LoadedDataset) -> None:
    """Trim header whitespace and make column names unique.

    Header problems are recorded as notes; the schema check turns them into findings.
    """
    seen: dict[str, int] = {}
    new_columns: list[str] = []

    for index, raw in enumerate(dataset.frame.columns):
        name = str(raw)
        stripped = name.strip()
        if stripped != name:
            dataset.notes.append(f"Column header {index + 1} had surrounding whitespace.")
        if not stripped or _UNNAMED_RE.match(stripped):
            stripped = f"column_{index + 1}"
            dataset.notes.append(f"Column header {index + 1} was empty and was auto-named.")

        if stripped in seen:
            seen[stripped] += 1
            unique = f"{stripped}_{seen[stripped]}"
            dataset.notes.append(f"Duplicate column header '{stripped}' was renamed to '{unique}'.")
            stripped = unique
        else:
            seen[stripped] = 0
        new_columns.append(stripped)

    dataset.frame.columns = pd.Index(new_columns)
    dataset.frame.reset_index(drop=True, inplace=True)
