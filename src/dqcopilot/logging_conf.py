"""Logging configuration with a defensive redaction filter.

The application deliberately never logs cell values from uploaded datasets. This module
adds a second line of defence: a filter that scrubs anything resembling an API key or an
email address from formatted log records.
"""

from __future__ import annotations

import json
import logging
import re
import sys
from typing import Any

_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_SECRET_RE = re.compile(r"sk-[A-Za-z0-9_\-]{8,}")

_CONFIGURED = False


class RedactionFilter(logging.Filter):
    """Redact API keys and email addresses from log messages."""

    def filter(self, record: logging.LogRecord) -> bool:  # noqa: D102 - inherited
        message = record.getMessage()
        redacted = _SECRET_RE.sub("[REDACTED_KEY]", message)
        redacted = _EMAIL_RE.sub("[REDACTED_EMAIL]", redacted)
        if redacted != message:
            record.msg = redacted
            record.args = ()
        return True


class JsonFormatter(logging.Formatter):
    """Minimal JSON line formatter (no external dependency)."""

    _RESERVED = frozenset(logging.LogRecord("", 0, "", 0, "", (), None).__dict__) | {
        "message",
        "asctime",
        "taskName",
    }

    def format(self, record: logging.LogRecord) -> str:  # noqa: D102 - inherited
        payload: dict[str, Any] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in self._RESERVED and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging(level: str = "INFO", json_output: bool = False) -> None:
    """Configure the root logger once per process.

    Args:
        level: Logging level name, e.g. ``"INFO"``.
        json_output: Emit JSON lines instead of human readable text.
    """
    global _CONFIGURED
    if _CONFIGURED:
        logging.getLogger().setLevel(level)
        return

    handler = logging.StreamHandler(sys.stderr)
    handler.addFilter(RedactionFilter())
    if json_output:
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)-8s %(name)s :: %(message)s")
        )

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a module level logger."""
    return logging.getLogger(name)
