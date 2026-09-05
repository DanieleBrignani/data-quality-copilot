"""Streamlit session-state helpers.

Keeping session access in one place means the page modules never touch raw
``st.session_state`` keys, which keeps the UI easy to reason about and to change.
"""

from __future__ import annotations

from typing import Any

import streamlit as st

from dqcopilot.ai import AiSuggestions
from dqcopilot.services.review import CleanedDataset, ReviewSession

_REVIEW = "dq_review"
_UPLOAD = "dq_upload_signature"
_ERROR = "dq_error"
_CLEANED = "dq_cleaned"
_AI = "dq_ai"
_PERSISTENCE = "dq_persistence_notes"


def get_review() -> ReviewSession | None:
    """Return the review session currently held, if any."""
    value = st.session_state.get(_REVIEW)
    return value if isinstance(value, ReviewSession) else None


def set_review(review: ReviewSession) -> None:
    """Store a review session and clear anything derived from a previous one."""
    st.session_state[_REVIEW] = review
    st.session_state[_ERROR] = None
    st.session_state.pop(_CLEANED, None)
    st.session_state.pop(_AI, None)


def clear_all() -> None:
    """Remove every piece of session state owned by this app."""
    for key in (_REVIEW, _UPLOAD, _ERROR, _CLEANED, _AI, _PERSISTENCE):
        st.session_state.pop(key, None)


def get_error() -> str | None:
    """Return the last ingestion error message, if any."""
    value = st.session_state.get(_ERROR)
    return value if isinstance(value, str) else None


def set_error(message: str) -> None:
    """Record an ingestion error and drop the stale review."""
    st.session_state[_ERROR] = message
    st.session_state.pop(_REVIEW, None)
    st.session_state.pop(_CLEANED, None)
    st.session_state.pop(_AI, None)


def upload_signature() -> str | None:
    """Return the signature of the upload behind the current review."""
    value = st.session_state.get(_UPLOAD)
    return value if isinstance(value, str) else None


def set_upload_signature(signature: str) -> None:
    """Record which upload the current review belongs to."""
    st.session_state[_UPLOAD] = signature


def get_cleaned() -> CleanedDataset | None:
    """Return the cleaned dataset, if corrections have been applied."""
    value = st.session_state.get(_CLEANED)
    return value if isinstance(value, CleanedDataset) else None


def set_cleaned(cleaned: CleanedDataset) -> None:
    """Store the cleaned dataset."""
    st.session_state[_CLEANED] = cleaned


def drop_cleaned() -> None:
    """Discard a stale cleaned dataset after the decisions changed."""
    st.session_state.pop(_CLEANED, None)


def get_ai() -> AiSuggestions | None:
    """Return the AI suggestions, if they have been requested."""
    value = st.session_state.get(_AI)
    return value if isinstance(value, AiSuggestions) else None


def set_ai(suggestions: AiSuggestions) -> None:
    """Store the AI suggestions."""
    st.session_state[_AI] = suggestions


def add_persistence_note(note: str) -> None:
    """Record a message about the database being unavailable."""
    notes: list[str] = st.session_state.setdefault(_PERSISTENCE, [])
    if note not in notes:
        notes.append(note)


def persistence_notes() -> list[str]:
    """Return recorded persistence problems."""
    value = st.session_state.get(_PERSISTENCE)
    return list(value) if isinstance(value, list) else []


def remember(key: str, value: Any) -> None:
    """Store an arbitrary value under a namespaced session key."""
    st.session_state[f"dq_{key}"] = value


def recall(key: str, default: Any = None) -> Any:
    """Read a value previously stored with :func:`remember`."""
    return st.session_state.get(f"dq_{key}", default)
