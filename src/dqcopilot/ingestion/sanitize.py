"""Filename and file-content sanitisation helpers.

These functions implement the security rules of the ingestion layer:

* only ``.csv`` and ``.xlsx`` are accepted;
* the declared extension must match the actual bytes (magic-number check);
* filenames are stripped of directory components and unsafe characters.
"""

from __future__ import annotations

import re
from pathlib import PurePath

from dqcopilot.ingestion.errors import UnsupportedFileTypeError

ALLOWED_EXTENSIONS: frozenset[str] = frozenset({".csv", ".xlsx"})
MAX_FILENAME_LENGTH = 120

_ZIP_MAGIC = b"PK\x03\x04"
_OLE2_MAGIC = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"  # legacy .xls / .doc container

_UNSAFE_CHARS = re.compile(r"[^A-Za-z0-9._-]+")
_REPEATED_DOTS = re.compile(r"\.{2,}")


def sanitize_filename(filename: str, fallback: str = "upload") -> str:
    """Return a safe, flat filename derived from ``filename``.

    Directory components, path traversal sequences and unusual characters are removed.
    The result is never empty and never starts with a dot.

    Args:
        filename: The user supplied name, possibly containing a path.
        fallback: Stem used when nothing usable remains.

    Returns:
        A filename safe to use for a temporary or exported file.
    """
    raw = (filename or "").replace("\\", "/").strip()
    base = PurePath(raw).name
    base = _REPEATED_DOTS.sub(".", base)
    base = _UNSAFE_CHARS.sub("_", base).strip("._")

    if not base:
        return fallback

    stem, _, suffix = base.rpartition(".")
    if not stem:
        stem, suffix = base, ""

    stem = stem[:MAX_FILENAME_LENGTH].strip("_-.") or fallback
    return f"{stem}.{suffix}" if suffix else stem


def get_extension(filename: str) -> str:
    """Return the lowercase extension of ``filename`` including the dot."""
    return PurePath(filename.replace("\\", "/")).suffix.lower()


def validate_extension(filename: str) -> str:
    """Validate that ``filename`` has a supported extension.

    Args:
        filename: Name of the uploaded file.

    Returns:
        The normalised extension, e.g. ``".csv"``.

    Raises:
        UnsupportedFileTypeError: If the extension is missing or not allowed.
    """
    extension = get_extension(filename)
    if extension not in ALLOWED_EXTENSIONS:
        allowed = ", ".join(sorted(ALLOWED_EXTENSIONS))
        raise UnsupportedFileTypeError(
            f"Unsupported file type '{extension or '(none)'}'. Allowed types: {allowed}."
        )
    return extension


def validate_content_matches_extension(content: bytes, extension: str) -> None:
    """Check that the file bytes are consistent with the declared extension.

    This blocks a renamed binary (for example ``payload.exe`` renamed to ``data.csv``)
    and legacy ``.xls``/macro containers disguised as ``.xlsx``.

    Args:
        content: The raw uploaded bytes.
        extension: The validated extension from :func:`validate_extension`.

    Raises:
        UnsupportedFileTypeError: If the content does not match the extension.
    """
    header = content[:8]

    if extension == ".xlsx":
        if header.startswith(_OLE2_MAGIC):
            raise UnsupportedFileTypeError(
                "This looks like a legacy .xls (or macro-enabled) workbook renamed to .xlsx. "
                "Please re-save it as a real .xlsx file."
            )
        if not header.startswith(_ZIP_MAGIC):
            raise UnsupportedFileTypeError(
                "The file does not look like a valid .xlsx workbook (missing ZIP signature)."
            )
        return

    # CSV: reject anything that is clearly a binary container.
    if header.startswith((_ZIP_MAGIC, _OLE2_MAGIC)) or b"\x00" in content[:4096]:
        raise UnsupportedFileTypeError(
            "The file is declared as .csv but contains binary data. Upload a plain text CSV."
        )
