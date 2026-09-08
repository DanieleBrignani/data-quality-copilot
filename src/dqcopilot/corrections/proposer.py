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

from dqcopilot.encoding_repair import repair_mojibake
from dqcopilot.models.corrections import (
    MAX_PREVIEW_ROWS,
    ChangePreview,
    CorrectionProposal,
    make_proposal_id,
)
from dqcopilot.models.enums import CorrectionAction, IssueType, SemanticType
from dqcopilot.models.findings import Finding
from dqcopilot.models.profile import DatasetProfile
from dqcopilot.naming import looks_like_code, looks_like_coordinate
from dqcopilot.profiling.dependencies import Dependency, build_lookup, find_determinant
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

    # A value that another column already decides is recovered, never invented - and
    # when the determinant is missing too, saying so beats offering an average.
    dependency = find_determinant(frame, column)
    if dependency is not None:
        return _fill_from_related(finding, frame, profile, dependency)

    fill_value, strategy = _imputation_value(frame[column], column_profile.semantic_type, column)
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


def _imputation_value(
    series: pd.Series,
    semantic_type: SemanticType,
    column_name: str = "",
) -> tuple[object | None, str]:
    """Pick a defensible fill value, or ``None`` when none exists.

    Codes and coordinates are refused outright. A missing district is not the median
    district, a missing postcode is not the most common postcode, and a missing longitude
    is not the middle of the map: those answers are arithmetically valid and factually
    wrong. Unlike a biased average, they are wrong about one specific row, in a way a
    reader can look up. Such a column goes to manual review instead.
    """
    present = series[~missing_mask(series)]
    if present.empty:
        return None, ""

    unimputable = (
        looks_like_code(column_name)
        or looks_like_coordinate(column_name)
        or semantic_type is SemanticType.IDENTIFIER
    )
    if unimputable:
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


@builder_for(IssueType.CORRUPTED_ENCODING)
def _repair_encoding(
    finding: Finding, frame: pd.DataFrame, profile: DatasetProfile
) -> list[CorrectionProposal]:
    """Offer to decode mojibake back to the text the file really contained.

    This is the rare correction that restores information instead of trading it away:
    the repaired string is derived from the bytes already in the cell, not guessed from
    the rest of the column. Values whose bytes were already discarded are left alone -
    they appear in the finding and get no button, because nothing here can bring them
    back.
    """
    column = finding.column
    if column is None or column not in frame.columns:
        return []

    before = to_clean_strings(frame[column])
    mapping: dict[str, str] = {}
    for value in before.dropna().unique():
        text = str(value)
        repaired = repair_mojibake(text)
        if repaired is not None:
            mapping[text] = repaired

    if not mapping:
        return _manual_review(finding, frame, profile)

    after = before.map(lambda value: mapping.get(value, value), na_action="ignore")

    return [
        CorrectionProposal(
            proposal_id=make_proposal_id(finding.finding_id, CorrectionAction.REPAIR_ENCODING),
            finding_id=finding.finding_id,
            action=CorrectionAction.REPAIR_ENCODING,
            column=column,
            title=f"Repair the mangled characters in '{column}'",
            description=(
                f"Decode {len(mapping):,} damaged value(s) in '{column}' back to the text the "
                "source file contained. The mangled characters are the original UTF-8 bytes "
                "read through the wrong codec, so the repair is reversible arithmetic on "
                "those bytes rather than a guess about what the word should have been."
            ),
            rationale=finding.explanation,
            parameters={"mapping": dict(list(mapping.items())[:200])},
            affected_rows=int((before != after).sum()),
            preview=_preview(column, before, after),
            confidence=0.95,
        )
    ]


@builder_for(IssueType.PLACEHOLDER_VALUE)
def _clear_placeholders(
    finding: Finding, frame: pd.DataFrame, profile: DatasetProfile
) -> list[CorrectionProposal]:
    """Offer to turn filler text into a genuine missing value."""
    column = finding.column
    if column is None or column not in frame.columns:
        return []

    values = [str(value) for value in finding.details.get("values", [])]
    if not values:
        return _manual_review(finding, frame, profile)

    before = to_clean_strings(frame[column])
    targets = set(values)
    matched = before.str.strip().str.casefold().isin(targets)
    matched = matched.astype("boolean").fillna(False).astype(bool)
    after = before.mask(matched, other=pd.NA)

    return [
        CorrectionProposal(
            proposal_id=make_proposal_id(finding.finding_id, CorrectionAction.CLEAR_INVALID_VALUES),
            finding_id=finding.finding_id,
            action=CorrectionAction.CLEAR_INVALID_VALUES,
            column=column,
            title=f"Blank the placeholder values in '{column}'",
            description=(
                f"Replace {finding.affected_rows:,} filler value(s) in '{column}' with an empty "
                "cell, so the gap is counted as missing instead of passing for data. This "
                "lowers the completeness of the column on purpose: the information was never "
                "there, and a report that says so is the accurate one."
            ),
            rationale=finding.explanation,
            parameters={"predicate": "placeholder_value", "values": sorted(targets)},
            affected_rows=finding.affected_rows,
            preview=_preview(column, before, after),
            destructive=True,
        )
    ]


def _fill_from_related(
    finding: Finding,
    frame: pd.DataFrame,
    profile: DatasetProfile,
    dependency: Dependency,
) -> list[CorrectionProposal]:
    """Offer a lookup instead of an average, or explain why neither is possible.

    Where the determining column is populated, the gap has one correct answer recorded
    elsewhere in the file. Where it is missing too, nothing here can supply the value -
    and a proposal that says so is worth more than one that fills the column with a
    number nobody should trust.
    """
    column = finding.column
    if column is None or column not in frame.columns:
        return []

    if dependency.recoverable_rows == 0:
        return _unrecoverable(finding, frame, profile, dependency, column)

    gaps = missing_mask(frame[column])
    keys = to_clean_strings(frame[dependency.determinant]).astype("string").str.strip()
    # Only the keys the gaps actually need travel with the proposal. A full lookup table
    # would have to be truncated on a large column, and a truncated table silently fills
    # fewer rows than the proposal promises.
    needed = set(keys[gaps].dropna())
    lookup = {key: value for key, value in build_lookup(frame, dependency).items() if key in needed}

    before = to_clean_strings(frame[column])
    supplied = keys.map(lambda key: lookup.get(key), na_action="ignore").astype("string")
    after = before.mask(gaps & supplied.notna(), other=supplied)

    remaining = (
        f" The remaining {dependency.unrecoverable_rows:,} gap(s) stay empty, because "
        f"'{dependency.determinant}' is missing there too."
        if dependency.unrecoverable_rows
        else ""
    )

    return [
        CorrectionProposal(
            proposal_id=make_proposal_id(finding.finding_id, CorrectionAction.FILL_FROM_RELATED),
            finding_id=finding.finding_id,
            action=CorrectionAction.FILL_FROM_RELATED,
            column=column,
            title=f"Recover '{column}' from '{dependency.determinant}'",
            description=(
                f"Fill {dependency.recoverable_rows:,} gap(s) in '{column}' by looking the "
                f"value up in '{dependency.determinant}', which decides it: across the "
                f"{dependency.evidence_rows:,} rows where both are populated, each of the "
                f"{dependency.distinct_keys:,} keys maps to exactly one value. This recovers "
                "what the file already records instead of inventing a plausible number." + remaining
            ),
            rationale=finding.explanation,
            parameters={"determinant": dependency.determinant, "mapping": lookup},
            affected_rows=dependency.recoverable_rows,
            preview=_preview(column, before, after),
            destructive=False,
            confidence=0.9,
        )
    ]


def _unrecoverable(
    finding: Finding,
    frame: pd.DataFrame,
    profile: DatasetProfile,
    dependency: Dependency,
    column: str,
) -> list[CorrectionProposal]:
    """Explain that the value is knowable in principle but absent from this file."""
    review = _manual_review(finding, frame, profile)
    return [
        review[0].model_copy(
            update={
                "description": (
                    f"'{column}' is not a quantity to average: its value is decided by "
                    f"'{dependency.determinant}', which agrees with it on all "
                    f"{dependency.evidence_rows:,} rows where both are filled in. That "
                    "would make the gaps recoverable by lookup - except that "
                    f"'{dependency.determinant}' is missing on every one of the "
                    f"{dependency.unrecoverable_rows:,} affected rows as well. Nothing in "
                    "this file can supply the value; it has to come from the source system "
                    "or stay empty."
                )
            }
        )
    ]
