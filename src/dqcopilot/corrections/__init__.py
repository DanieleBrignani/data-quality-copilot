"""Correction proposal and approval workflow.

Proposals are built by :mod:`dqcopilot.corrections.proposer` and applied by
:mod:`dqcopilot.corrections.applier`. Nothing is ever applied without an explicit
approval decision.
"""

from dqcopilot.corrections.applier import (
    APPLICATION_ORDER,
    CorrectionError,
    apply_corrections,
    approved_proposals,
)
from dqcopilot.corrections.proposer import propose_corrections

__all__ = [
    "APPLICATION_ORDER",
    "CorrectionError",
    "apply_corrections",
    "approved_proposals",
    "propose_corrections",
]
