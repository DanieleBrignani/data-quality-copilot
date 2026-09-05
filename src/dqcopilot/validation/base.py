"""Base types for deterministic data quality checks.

A *check* is a small, pure class that receives a :class:`CheckContext` and returns a
list of :class:`~dqcopilot.models.findings.Finding`. Checks never mutate the frame and
never call an external service, which is what makes them reproducible and cheap to test.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import pandas as pd

from dqcopilot.models.enums import Severity
from dqcopilot.models.findings import MAX_ROW_SAMPLE, Finding
from dqcopilot.models.profile import ColumnProfile, DatasetProfile
from dqcopilot.naming import column_tokens, name_matches

__all__ = [
    "Check",
    "CheckContext",
    "column_tokens",
    "name_matches",
    "pct",
    "ratio_severity",
    "sample_indices",
]

if TYPE_CHECKING:  # pragma: no cover - import cycle guard
    from dqcopilot.rules.models import BusinessRuleSet


@dataclass(slots=True)
class CheckContext:
    """Everything a check needs to inspect a dataset."""

    frame: pd.DataFrame
    profile: DatasetProfile
    rules: BusinessRuleSet | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def row_count(self) -> int:
        """Number of rows in the dataset."""
        return int(self.frame.shape[0])

    def columns(self) -> list[ColumnProfile]:
        """Column profiles in file order."""
        return self.profile.columns

    def series(self, name: str) -> pd.Series:
        """Return the column ``name`` from the frame."""
        return self.frame[name]


class Check(abc.ABC):
    """Abstract base class for a deterministic check."""

    #: Stable identifier, used in finding ids and in the report.
    check_id: str = "check"
    #: Human readable name shown in the UI.
    title: str = "Check"
    #: One line description of what the check looks for.
    description: str = ""

    @abc.abstractmethod
    def run(self, context: CheckContext) -> list[Finding]:
        """Inspect the dataset and return zero or more findings."""
        raise NotImplementedError


def sample_indices(mask: pd.Series, limit: int = MAX_ROW_SAMPLE) -> list[int]:
    """Return up to ``limit`` positional row indices where ``mask`` is True."""
    if mask.empty:
        return []
    positions = mask.to_numpy().nonzero()[0]
    return [int(position) for position in positions[:limit]]


def ratio_severity(
    ratio: float,
    critical_above: float = 0.5,
    high_above: float = 0.2,
    medium_above: float = 0.05,
) -> Severity:
    """Map an affected-row ratio to a severity level.

    The thresholds are deliberately simple and are documented in the README so a
    reviewer can see exactly why a finding got its severity.
    """
    if ratio > critical_above:
        return Severity.CRITICAL
    if ratio > high_above:
        return Severity.HIGH
    if ratio > medium_above:
        return Severity.MEDIUM
    return Severity.LOW


def pct(value: float) -> str:
    """Format a 0-1 ratio as a short percentage string."""
    return f"{value * 100:.1f}%"
