"""Turn findings into concrete, previewable correction proposals.

Two rules govern this module:

1. **Nothing is applied here.** A proposal describes a change and shows a before/after
   preview; only :mod:`dqcopilot.corrections.applier` mutates data, and only for
   proposals the user approved.
2. **Ambiguity is never resolved silently.** Where a fix would require guessing - which
   of two duplicate records to keep, what a missing revenue figure should be, which
   spelling of a country is canonical when both are equally frequent - the proposal is
   :data:`~dqcopilot.models.enums.CorrectionAction.MANUAL_REVIEW`: it explains the
   problem and asks for a human decision instead of offering a button.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import pandas as pd

from dqcopilot.models.corrections import (
    MAX_PREVIEW_ROWS,
    ChangePreview,
    CorrectionProposal,
    make_proposal_id,
)
from dqcopilot.models.enums import CorrectionAction, IssueType, SemanticType
from dqcopilot.models.findings import Finding
from dqcopilot.models.profile import DatasetProfile
from dqcopilot.profiling.type_inference import (
    coerce_numeric,
    missing_mask,
    parse_dates_with_format,
    to_clean_strings,
)

ProposalBuilder = Callable[[Finding, pd.DataFrame, DatasetProfile], list[CorrectionProposal]]

_BUILDERS: dict[IssueType, ProposalBuilder] = {}


def builder_for(issue_type: IssueType) -> Callable[[ProposalBuilder], ProposalBuilder]:
    """Register a proposal builder for an issue type."""

    def decorate(function: ProposalBuilder) -> ProposalBuilder:
        _BUILDERS[issue_type] = function
        return function

    return decorate


def propose_corrections(
    findings: list[Finding],
    frame: pd.DataFrame,
    profile: DatasetProfile,
) -> list[CorrectionProposal]:
    """Build a correction proposal for every finding that has one.

    Args:
        findings: The findings to consider.
        frame: The dataset the findings came from.
        profile: The dataset profile.

    Returns:
        Proposals in the same order as ``findings``, deduplicated by proposal id.
    """
    proposals: list[CorrectionProposal] = []
    seen: set[str] = set()

    for finding in findings:
        builder = _BUILDERS.get(finding.issue_type, _manual_review)
        for proposal in builder(finding, frame, profile):
            if proposal.proposal_id in seen:
                continue
            seen.add(proposal.proposal_id)
            proposals.append(proposal)

    return proposals


# --------------------------------------------------------------------------- helpers


def _preview(
    column: str,
    before: pd.Series,
    after: pd.Series,
    limit: int = MAX_PREVIEW_ROWS,
) -> list[ChangePreview]:
    """Build a before/after preview from two aligned Series."""
    changed = before.astype("string").fillna("") != after.astype("string").fillna("")
    rows = list(before.index[changed.to_numpy()][:limit])
    return [
        ChangePreview(
            row_index=int(row),
            column=column,
            before=_render(before.loc[row]),
            after=_render(after.loc[row]),
        )
        for row in rows
    ]


def _render(value: object) -> str:
    """Render a cell for display, making empty values and invisible padding visible."""
    if value is None:
        return "(empty)"

    if isinstance(value, str):
        if not value:
            return "(empty)"
        # Square brackets are the only way to show padding on screen.
        return f"[{value}]" if value != value.strip() else value

    if value is pd.NaT or value is pd.NA:
        return "(empty)"
    if isinstance(value, float) and math.isnan(value):
        return "(empty)"
    return str(value)


def _manual_review(
    finding: Finding,
    frame: pd.DataFrame,  # noqa: ARG001 - part of the builder signature
    profile: DatasetProfile,  # noqa: ARG001 - part of the builder signature
) -> list[CorrectionProposal]:
    """Fallback: describe the decision a human has to make."""
    return [
        CorrectionProposal(
            proposal_id=make_proposal_id(finding.finding_id, CorrectionAction.MANUAL_REVIEW),
            finding_id=finding.finding_id,
            action=CorrectionAction.MANUAL_REVIEW,
            source=finding.source,
            column=finding.column,
            title=f"Review: {finding.title}",
            description=(
                "No safe automatic fix exists for this issue. It needs a decision from "
                "someone who knows the data."
            ),
            rationale=finding.explanation,
            affected_rows=finding.affected_rows,
            confidence=finding.confidence,
        )
    ]


# --------------------------------------------------------------------------- builders


@builder_for(IssueType.LEADING_TRAILING_WHITESPACE)
def _strip_whitespace(
    finding: Finding, frame: pd.DataFrame, profile: DatasetProfile
) -> list[CorrectionProposal]:
    column = finding.column
    if column is None or column not in frame.columns:
        return []

    before = to_clean_strings(frame[column])
    after = before.str.strip()

    return [
        CorrectionProposal(
            proposal_id=make_proposal_id(finding.finding_id, CorrectionAction.STRIP_WHITESPACE),
            finding_id=finding.finding_id,
            action=CorrectionAction.STRIP_WHITESPACE,
            column=column,
            title=f"Trim whitespace in '{column}'",
            description=(
                f"Remove leading and trailing whitespace from {finding.affected_rows:,} "
                f"value(s) in '{column}'. The visible text is unchanged."
            ),
            rationale=finding.explanation,
            affected_rows=finding.affected_rows,
            preview=_preview(column, before, after),
        )
    ]


@builder_for(IssueType.INCONSISTENT_CAPITALIZATION)
def _normalize_case(
    finding: Finding, frame: pd.DataFrame, profile: DatasetProfile
) -> list[CorrectionProposal]:
    column = finding.column
    if column is None or column not in frame.columns:
        return []

    before = to_clean_strings(frame[column])
    mapping = _canonical_spelling_map(before, key=lambda value: value.strip().casefold())
    if not mapping:
        return []

    after = before.map(lambda value: mapping.get(value, value), na_action="ignore")
    ambiguous = _ambiguous_keys(before, key=lambda value: value.strip().casefold())

    return [
        CorrectionProposal(
            proposal_id=make_proposal_id(finding.finding_id, CorrectionAction.NORMALIZE_CASE),
            finding_id=finding.finding_id,
            action=CorrectionAction.NORMALIZE_CASE,
            column=column,
            title=f"Unify capitalisation in '{column}'",
            description=(
                f"Rewrite each value in '{column}' using the spelling that already occurs "
                "most often in the column. The most frequent form wins, so no new spelling "
                "is invented."
                + (
                    f" {len(ambiguous)} group(s) have no clear winner and are left untouched."
                    if ambiguous
                    else ""
                )
            ),
            rationale=finding.explanation,
            parameters={"mapping": dict(list(mapping.items())[:50])},
            affected_rows=int((before != after).sum()),
            preview=_preview(column, before, after),
        )
    ]


@builder_for(IssueType.INCONSISTENT_CATEGORY)
def _map_category(
    finding: Finding, frame: pd.DataFrame, profile: DatasetProfile
) -> list[CorrectionProposal]:
    import re

    column = finding.column
    if column is None or column not in frame.columns:
        return []

    def key(value: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", value.strip().casefold())

    before = to_clean_strings(frame[column])
    mapping = _canonical_spelling_map(before, key=key)
    if not mapping:
        return []

    after = before.map(lambda value: mapping.get(value, value), na_action="ignore")

    return [
        CorrectionProposal(
            proposal_id=make_proposal_id(finding.finding_id, CorrectionAction.MAP_CATEGORY),
            finding_id=finding.finding_id,
            action=CorrectionAction.MAP_CATEGORY,
            column=column,
            title=f"Merge equivalent categories in '{column}'",
            description=(
                f"Collapse values in '{column}' that differ only by punctuation or spacing "
                "onto the spelling that occurs most often. Genuinely different words are "
                "never merged - only forms that are identical once punctuation is removed."
            ),
            rationale=finding.explanation,
            parameters={"mapping": dict(list(mapping.items())[:50])},
            affected_rows=int((before != after).sum()),
            preview=_preview(column, before, after),
        )
    ]


@builder_for(IssueType.NUMERIC_STORED_AS_TEXT)
def _cast_numeric(
    finding: Finding, frame: pd.DataFrame, profile: DatasetProfile
) -> list[CorrectionProposal]:
    column = finding.column
    if column is None or column not in frame.columns:
        return []

    before = to_clean_strings(frame[column])
    parsed = coerce_numeric(frame[column])
    unparseable = int((parsed.isna() & ~missing_mask(frame[column])).sum())
    after = parsed.map(lambda value: "" if pd.isna(value) else f"{value:g}")

    return [
        CorrectionProposal(
            proposal_id=make_proposal_id(finding.finding_id, CorrectionAction.CAST_TO_NUMERIC),
            finding_id=finding.finding_id,
            action=CorrectionAction.CAST_TO_NUMERIC,
            column=column,
            title=f"Convert '{column}' to numbers",
            description=(
                f"Strip currency symbols, thousands separators and percent signs from "
                f"'{column}' and store it as a real number."
                + (
                    f" Warning: {unparseable:,} value(s) cannot be parsed and would become "
                    "empty. Review them before approving."
                    if unparseable
                    else ""
                )
            ),
            rationale=finding.explanation,
            parameters={"unparseable_becomes_empty": unparseable},
            affected_rows=int(parsed.notna().sum()),
            preview=_preview(column, before, after),
            destructive=unparseable > 0,
        )
    ]


@builder_for(IssueType.INCONSISTENT_DATE_FORMAT)
def _parse_dates(
    finding: Finding, frame: pd.DataFrame, profile: DatasetProfile
) -> list[CorrectionProposal]:
    column = finding.column
    if column is None or column not in frame.columns:
        return []

    before = to_clean_strings(frame[column])
    parsed = parse_dates_with_format(frame[column], None)
    after = parsed.dt.strftime("%Y-%m-%d").fillna("")
    unparseable = int((parsed.isna() & ~missing_mask(frame[column])).sum())

    return [
        CorrectionProposal(
            proposal_id=make_proposal_id(finding.finding_id, CorrectionAction.PARSE_DATES),
            finding_id=finding.finding_id,
            action=CorrectionAction.PARSE_DATES,
            column=column,
            title=f"Rewrite '{column}' as ISO dates",
            description=(
                f"Parse '{column}' and rewrite every value as YYYY-MM-DD. Ambiguous values "
                "such as 03/04/2024 are read as day-first; check the preview carefully, "
                "because reading them month-first would give a different day."
                + (
                    f" {unparseable:,} value(s) cannot be parsed and would become empty."
                    if unparseable
                    else ""
                )
            ),
            rationale=finding.explanation,
            parameters={"output_format": "%Y-%m-%d", "dayfirst": True},
            affected_rows=int((before != after).sum()),
            preview=_preview(column, before, after),
            destructive=unparseable > 0,
            confidence=0.7,
        )
    ]


@builder_for(IssueType.EXACT_DUPLICATE_ROWS)
def _drop_duplicates(
    finding: Finding, frame: pd.DataFrame, profile: DatasetProfile
) -> list[CorrectionProposal]:
    return [
        CorrectionProposal(
            proposal_id=make_proposal_id(finding.finding_id, CorrectionAction.DROP_DUPLICATE_ROWS),
            finding_id=finding.finding_id,
            action=CorrectionAction.DROP_DUPLICATE_ROWS,
            column=None,
            title="Remove exact duplicate rows",
            description=(
                f"Keep the first occurrence of each repeated row and drop the other "
                f"{finding.affected_rows:,}. Rows are compared after trimming whitespace "
                "and ignoring case. This removes rows from the dataset."
            ),
            rationale=finding.explanation,
            parameters={"keep": "first"},
            affected_rows=finding.affected_rows,
            destructive=True,
        )
    ]


@builder_for(IssueType.INVALID_EMAIL)
def _clear_invalid_emails(
    finding: Finding, frame: pd.DataFrame, profile: DatasetProfile
) -> list[CorrectionProposal]:
    column = finding.column
    if column is None or column not in frame.columns:
        return []

    from dqcopilot.profiling.type_inference import EMAIL_RE

    before = to_clean_strings(frame[column])
    invalid = before.str.strip().map(
        lambda value: not bool(EMAIL_RE.match(value)), na_action="ignore"
    )
    invalid = invalid.astype("boolean").fillna(False).astype(bool)
    after = before.mask(invalid, other=pd.NA)

    return [
        CorrectionProposal(
            proposal_id=make_proposal_id(finding.finding_id, CorrectionAction.CLEAR_INVALID_VALUES),
            finding_id=finding.finding_id,
            action=CorrectionAction.CLEAR_INVALID_VALUES,
            column=column,
            title=f"Blank the invalid addresses in '{column}'",
            description=(
                f"Replace {finding.affected_rows:,} malformed address(es) in '{column}' with "
                "an empty value, so downstream systems treat them as missing rather than "
                "as a real address. The original text is lost - export the audit log first "
                "if you need it."
            ),
            rationale=finding.explanation,
            parameters={"predicate": "invalid_email"},
            affected_rows=finding.affected_rows,
            preview=_preview(column, before, after),
            destructive=True,
        )
    ]


@builder_for(IssueType.INVALID_DATE)
def _clear_invalid_dates(
    finding: Finding, frame: pd.DataFrame, profile: DatasetProfile
) -> list[CorrectionProposal]:
    column = finding.column
    if column is None or column not in frame.columns:
        return []

    before = to_clean_strings(frame[column])
    parsed = parse_dates_with_format(frame[column], None)
    invalid = parsed.isna() & ~missing_mask(frame[column])
    after = before.mask(invalid, other=pd.NA)

    return [
        CorrectionProposal(
            proposal_id=make_proposal_id(finding.finding_id, CorrectionAction.CLEAR_INVALID_VALUES),
            finding_id=finding.finding_id,
            action=CorrectionAction.CLEAR_INVALID_VALUES,
            column=column,
            title=f"Blank the impossible dates in '{column}'",
            description=(
                f"Replace {finding.affected_rows:,} unparseable date(s) in '{column}' with an "
                "empty value. A date like 30 February cannot be repaired without knowing "
                "what was meant, so the honest fix is to mark it missing."
            ),
            rationale=finding.explanation,
            parameters={"predicate": "invalid_date"},
            affected_rows=finding.affected_rows,
            preview=_preview(column, before, after),
            destructive=True,
        )
    ]


@builder_for(IssueType.MISSING_VALUES)
def _fill_missing(
    finding: Finding, frame: pd.DataFrame, profile: DatasetProfile
) -> list[CorrectionProposal]:
    column = finding.column
    if column is None or column not in frame.columns:
        return []

    column_profile = profile.column(column)
    if column_profile is None:
        return []

    # A column that is entirely empty has nothing to impute from.
    if column_profile.missing_count >= column_profile.row_count:
        return _manual_review(finding, frame, profile)

    fill_value, strategy = _imputation_value(frame[column], column_profile.semantic_type)
    if fill_value is None:
        return _manual_review(finding, frame, profile)

    before = to_clean_strings(frame[column])
    after = before.mask(missing_mask(frame[column]), other=str(fill_value))

    return [
        CorrectionProposal(
            proposal_id=make_proposal_id(finding.finding_id, CorrectionAction.FILL_MISSING),
            finding_id=finding.finding_id,
            action=CorrectionAction.FILL_MISSING,
            column=column,
            title=f"Fill the gaps in '{column}' with the {strategy}",
            description=(
                f"Replace {finding.affected_rows:,} missing value(s) in '{column}' with "
                f"'{fill_value}' (the {strategy} of the values that are present). "
                "This is imputation, not recovery: it invents data that was never "
                "collected, and it will bias any statistic computed on the column. "
                "Prefer leaving the gaps visible unless a downstream system requires a value."
            ),
            rationale=finding.explanation,
            parameters={"value": str(fill_value), "strategy": strategy},
            affected_rows=finding.affected_rows,
            preview=_preview(column, before, after),
            destructive=False,
            confidence=0.5,
        )
    ]


def _imputation_value(series: pd.Series, semantic_type: SemanticType) -> tuple[object | None, str]:
    """Pick a defensible fill value, or ``None`` when none exists."""
    present = series[~missing_mask(series)]
    if present.empty:
        return None, ""

    if semantic_type.is_numeric:
        numeric = coerce_numeric(present).dropna()
        if numeric.empty:
            return None, ""
        return round(float(numeric.median()), 6), "median"

    if semantic_type in (SemanticType.CATEGORICAL, SemanticType.BOOLEAN):
        counts = to_clean_strings(present).str.strip().value_counts()
        if counts.empty:
            return None, ""
        # Refuse to guess when the top two values are equally common.
        if len(counts) > 1 and counts.iloc[0] == counts.iloc[1]:
            return None, ""
        return str(counts.index[0]), "most frequent value"

    return None, ""


def _canonical_spelling_map(
    series: pd.Series,
    key: Callable[[str], str],
) -> dict[str, str]:
    """Map each written form to the most frequent form sharing its key.

    Groups where the two most frequent forms tie are omitted: with no majority there is
    no non-arbitrary winner, and picking one would be a guess dressed up as a fix.
    """
    values = series.dropna().astype(str)
    if values.empty:
        return {}

    counts = values.value_counts()
    groups: dict[str, list[str]] = {}
    for form in counts.index:
        groups.setdefault(key(str(form)), []).append(str(form))

    mapping: dict[str, str] = {}
    for forms in groups.values():
        if len(forms) < 2:
            continue
        ranked = sorted(forms, key=lambda form: (-int(counts[form]), form))
        if len(ranked) > 1 and counts[ranked[0]] == counts[ranked[1]]:
            continue  # ambiguous: no majority spelling
        winner = ranked[0]
        for form in forms:
            if form != winner:
                mapping[form] = winner
    return mapping


def _ambiguous_keys(series: pd.Series, key: Callable[[str], str]) -> list[str]:
    """Return group keys that have no majority spelling."""
    values = series.dropna().astype(str)
    if values.empty:
        return []

    counts = values.value_counts()
    groups: dict[str, list[str]] = {}
    for form in counts.index:
        groups.setdefault(key(str(form)), []).append(str(form))

    return [
        group_key
        for group_key, forms in groups.items()
        if len(forms) > 1 and counts[forms[0]] == counts[forms[1]]
    ]
