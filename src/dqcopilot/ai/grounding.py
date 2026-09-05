"""Check AI output against the dataset it claims to describe.

Schema validation proves a response is *well formed*. It does not prove it is *true*.
A model can return a perfectly valid mapping for a column that does not exist, or
propose merging a value that never appears in the data. Everything here answers the
second question, and anything that fails is dropped with a logged reason rather than
shown to the user.

This is what makes the AI suggestions safe to display next to deterministic findings:
by the time a suggestion reaches the screen it refers to real columns and real values,
and it still cannot change anything without an explicit approval.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from dqcopilot.ai.schemas import (
    SUPPORTED_RULE_TYPES,
    CategoryMappingSuggestion,
    ColumnInterpretation,
    SuggestedRule,
)
from dqcopilot.logging_conf import get_logger
from dqcopilot.models.profile import DatasetProfile
from dqcopilot.profiling.type_inference import missing_mask, to_clean_strings

logger = get_logger(__name__)


@dataclass(slots=True)
class GroundingReport[T]:
    """Which suggestions survived grounding, and why the others did not."""

    accepted: list[T] = field(default_factory=list)
    rejected: list[tuple[T, str]] = field(default_factory=list)

    @property
    def rejection_reasons(self) -> list[str]:
        """Human readable reasons, for the UI's transparency panel."""
        return [reason for _, reason in self.rejected]

    def reject(self, item: T, reason: str) -> None:
        """Record a rejected suggestion."""
        self.rejected.append((item, reason))
        logger.info("AI suggestion rejected", extra={"reason": reason})

    def accept(self, item: T) -> None:
        """Record an accepted suggestion."""
        self.accepted.append(item)


def column_values(series: pd.Series) -> set[str]:
    """Return the distinct trimmed values present in a column."""
    present = to_clean_strings(series)[~missing_mask(series)]
    return {str(value).strip() for value in present.unique()}


def ground_category_mappings(
    suggestions: list[CategoryMappingSuggestion],
    frame: pd.DataFrame,
    minimum_confidence: float = 0.6,
) -> GroundingReport[CategoryMappingSuggestion]:
    """Keep only mappings that refer to values actually present in the dataset.

    A mapping is accepted when the column exists, both values occur in it, they are not
    the same value, the mapping does not contradict another accepted mapping, and the
    model's own confidence clears the threshold.
    """
    report: GroundingReport[CategoryMappingSuggestion] = GroundingReport()
    targets: dict[tuple[str, str], str] = {}

    for suggestion in suggestions:
        column = suggestion.column
        if column not in frame.columns:
            report.reject(suggestion, f"Column '{column}' is not in the dataset.")
            continue
        if suggestion.confidence < minimum_confidence:
            report.reject(
                suggestion,
                f"Confidence {suggestion.confidence:.0%} is below the "
                f"{minimum_confidence:.0%} threshold.",
            )
            continue

        values = column_values(frame[column])
        source = suggestion.from_value.strip()
        target = suggestion.to_value.strip()

        if source == target:
            report.reject(suggestion, f"'{source}' maps onto itself.")
            continue
        if source not in values:
            report.reject(suggestion, f"'{source}' does not occur in column '{column}'.")
            continue
        if target not in values:
            report.reject(
                suggestion,
                f"'{target}' does not occur in column '{column}', so the mapping would "
                "invent a new value.",
            )
            continue

        key = (column, source)
        if key in targets:
            report.reject(suggestion, f"'{source}' already has a mapping in '{column}'.")
            continue
        # Reject cycles: mapping A->B and B->A cannot both be applied.
        if (column, target) in targets and targets[(column, target)] == source:
            report.reject(suggestion, f"'{source}' and '{target}' map to each other.")
            continue

        targets[key] = target
        report.accept(suggestion)

    return report


def ground_column_interpretations(
    interpretations: list[ColumnInterpretation],
    profile: DatasetProfile,
) -> GroundingReport[ColumnInterpretation]:
    """Keep only interpretations that name a real column, one per column."""
    report: GroundingReport[ColumnInterpretation] = GroundingReport()
    known = {column.name for column in profile.columns}
    seen: set[str] = set()

    for interpretation in interpretations:
        if interpretation.column not in known:
            report.reject(
                interpretation, f"Column '{interpretation.column}' is not in the dataset."
            )
            continue
        if interpretation.column in seen:
            report.reject(
                interpretation, f"Duplicate interpretation for '{interpretation.column}'."
            )
            continue
        seen.add(interpretation.column)
        report.accept(interpretation)

    return report


def ground_suggested_rules(
    rules: list[SuggestedRule],
    frame: pd.DataFrame,
    minimum_confidence: float = 0.6,
) -> GroundingReport[SuggestedRule]:
    """Keep only rules that are well formed, supported, and about real columns.

    A suggested rule is also required to be *consistent with the data it describes*: a
    range rule whose bounds are inverted, or an allowed-values rule listing values that
    never occur, is dropped.
    """
    report: GroundingReport[SuggestedRule] = GroundingReport()
    seen: set[str] = set()

    for rule in rules:
        if rule.rule_type not in SUPPORTED_RULE_TYPES:
            report.reject(rule, f"Rule type '{rule.rule_type}' is not supported.")
            continue
        if rule.column not in frame.columns:
            report.reject(rule, f"Column '{rule.column}' is not in the dataset.")
            continue
        if rule.confidence < minimum_confidence:
            report.reject(rule, f"Confidence {rule.confidence:.0%} is below the threshold.")
            continue
        if rule.name in seen:
            report.reject(rule, f"Duplicate rule name '{rule.name}'.")
            continue

        if rule.rule_type == "range":
            if rule.minimum is None and rule.maximum is None:
                report.reject(rule, "A range rule needs at least one bound.")
                continue
            if (
                rule.minimum is not None
                and rule.maximum is not None
                and rule.minimum > rule.maximum
            ):
                report.reject(rule, "The range rule has an inverted interval.")
                continue

        if rule.rule_type == "allowed_values":
            if not rule.allowed_values:
                report.reject(rule, "An allowed_values rule needs a vocabulary.")
                continue
            present = {value.casefold() for value in column_values(frame[rule.column])}
            overlap = {value.strip().casefold() for value in rule.allowed_values} & present
            if not overlap:
                report.reject(
                    rule,
                    f"None of the proposed values for '{rule.column}' occur in the data.",
                )
                continue

        seen.add(rule.name)
        report.accept(rule)

    return report
