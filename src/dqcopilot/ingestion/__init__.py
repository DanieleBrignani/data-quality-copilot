"""File ingestion: validation, sanitisation and parsing of CSV/XLSX uploads."""

from dqcopilot.ingestion.errors import (
    CorruptFileError,
    DatasetTooLargeError,
    EmptyFileError,
    FileTooLargeError,
    IngestionError,
    UnsupportedFileTypeError,
)
from dqcopilot.ingestion.loader import LoadedDataset, load_tabular
from dqcopilot.ingestion.sanitize import (
    ALLOWED_EXTENSIONS,
    sanitize_filename,
    validate_content_matches_extension,
    validate_extension,
)

__all__ = [
    "ALLOWED_EXTENSIONS",
    "CorruptFileError",
    "DatasetTooLargeError",
    "EmptyFileError",
    "FileTooLargeError",
    "IngestionError",
    "LoadedDataset",
    "UnsupportedFileTypeError",
    "load_tabular",
    "sanitize_filename",
    "validate_content_matches_extension",
    "validate_extension",
]
