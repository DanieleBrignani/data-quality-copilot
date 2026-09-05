"""Typed ingestion errors.

Every error carries a message that is safe and useful to show directly in the UI:
it explains what was rejected and what the user should do, without echoing file content.
"""

from __future__ import annotations


class IngestionError(Exception):
    """Base class for all ingestion failures."""


class UnsupportedFileTypeError(IngestionError):
    """The file extension or the actual content is not a supported tabular format."""


class FileTooLargeError(IngestionError):
    """The upload exceeds the configured maximum size."""


class EmptyFileError(IngestionError):
    """The file contains no bytes, no rows or no columns."""


class CorruptFileError(IngestionError):
    """The file could not be parsed as CSV or XLSX."""


class DatasetTooLargeError(IngestionError):
    """The parsed dataset exceeds the configured row or column limits."""
