"""Deterministic data quality validation.

Importing this package imports every check module, which registers the checks with the
registry as a side effect.
"""

from dqcopilot.validation import checks as _checks  # noqa: F401  (import registers checks)
from dqcopilot.validation.base import Check, CheckContext, ratio_severity, sample_indices
from dqcopilot.validation.registry import register_check, registered_checks, run_checks

__all__ = [
    "Check",
    "CheckContext",
    "ratio_severity",
    "register_check",
    "registered_checks",
    "run_checks",
    "sample_indices",
]
