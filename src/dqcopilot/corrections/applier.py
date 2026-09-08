"""Apply approved corrections to a dataset.

This is the only module in the project that changes data, and it changes nothing that
was not explicitly approved. It never mutates the input frame: a copy is returned, so
the original stays available for the before/after comparison and the audit log.
"""

from __future__ import annotations

import pandas as pd

from dqcopilot.logging_conf import get_logger
from dqcopilot.models.corrections import AppliedChange, CorrectionDecision, CorrectionProposal
from dqcopilot.models.enums import CorrectionAction, DecisionStatus
from dqcopilot.profiling.type_inference import (
    EMAIL_RE,
    coerce_numeric,
    missing_mask,
    parse_dates_with_format,
    to_clean_strings,
)

logger = get_logger(__name__)

#: Corrections are applied in this order regardless of the order they were approved in.
#: Cosmetic fixes run first so that de-duplication (which removes rows, and runs last)
#: sees already-normalised values. A different order would give a different result.
APPLICATION_ORDER: tuple[CorrectionAction, ...] = (
    # Encoding repair comes first: every later step compares text, and a corrupted
    # spelling would otherwise be treated as a category of its own.
    CorrectionAction.REPAIR_ENCODING,
    CorrectionAction.STRIP_WHITESPACE,
    CorrectionAction.NORMALIZE_CASE,
    CorrectionAction.MAP_CATEGORY,
    CorrectionAction.CAST_TO_NUMERIC,
    CorrectionAction.PARSE_DATES,
    CorrectionAction.CLEAR_INVALID_VALUES,
    CorrectionAction.CLIP_TO_RANGE,
    # Recovering a value from another column comes before inventing one from the
    # column's own distribution: a looked-up answer is right, not merely plausible.
    CorrectionAction.FILL_FROM_RELATED,
    CorrectionAction.FILL_MISSING,
    CorrectionAction.DROP_DUPLICATE_ROWS,
)


class CorrectionError(Exception):
    """A correction could not be applied to this dataset."""


def approved_proposals(
    proposals: list[CorrectionProposal],
    decisions: dict[str, CorrectionDecision],
) -> list[CorrectionProposal]:
    """Return the proposals the user approved, in application order.

    Args:
        proposals: All proposals offered.
        decisions: Decisions keyed by ``proposal_id``. Anything not present, or not
            explicitly approved, is treated as rejected.
    """
    approved = [
        proposal
        for proposal in proposals
        if decisions.get(proposal.proposal_id) is not None
        and decisions[proposal.proposal_id].status is DecisionStatus.APPROVED
        and proposal.action.is_applicable
    ]
    order = {action: index for index, action in enumerate(APPLICATION_ORDER)}
    return sorted(
        approved, key=lambda proposal: (order.get(proposal.action, 99), proposal.column or "")
    )


def apply_corrections(
    frame: pd.DataFrame,
    proposals: list[CorrectionProposal],
    decisions: dict[str, CorrectionDecision],
) -> tuple[pd.DataFrame, list[AppliedChange]]:
    """Apply every approved correction and return the cleaned dataset.

    Args:
        frame: The original dataset. It is copied, never modified.
        proposals: All proposals that were offered.
        decisions: The user's decisions, keyed by proposal id.

    Returns:
        A tuple of the cleaned frame and the list of changes actually applied.
    """
    cleaned = frame.copy()
    applied: list[AppliedChange] = []

    for proposal in approved_proposals(proposals, decisions):
        try:
            cleaned, rows_changed = _apply_one(cleaned, proposal)
        except Exception:  # noqa: BLE001 - one bad correction must not lose the others
            logger.exception(
                "Correction failed",
                extra={"proposal_id": proposal.proposal_id, "action": proposal.action.value},
            )
            continue

        if rows_changed == 0:
            continue

        applied.append(
            AppliedChange(
                proposal_id=proposal.proposal_id,
                action=proposal.action,
                column=proposal.column,
                rows_changed=rows_changed,
                parameters=proposal.parameters,
            )
        )
        logger.info(
            "Correction applied",
            extra={
                "proposal_id": proposal.proposal_id,
                "action": proposal.action.value,
                "column": proposal.column,
                "rows_changed": rows_changed,
            },
        )

    return cleaned, applied


# --------------------------------------------------------------------------- dispatch


def _apply_one(frame: pd.DataFrame, proposal: CorrectionProposal) -> tuple[pd.DataFrame, int]:
    """Apply one proposal, returning the new frame and the number of rows changed."""
    action = proposal.action

    if action is CorrectionAction.DROP_DUPLICATE_ROWS:
        return _drop_duplicates(frame)
    if action is CorrectionAction.FILL_FROM_RELATED:
        # The only column correction that reads a second column, so it needs the frame.
        return _fill_from_related(frame, proposal)

    column = proposal.column
    if column is None:
        raise CorrectionError(f"{action.value} needs a column but none was given.")
    if column not in frame.columns:
        raise CorrectionError(f"Column '{column}' is not present in the dataset.")

    handlers = {
        CorrectionAction.STRIP_WHITESPACE: _strip_whitespace,
        CorrectionAction.NORMALIZE_CASE: _apply_mapping,
        CorrectionAction.MAP_CATEGORY: _apply_mapping,
        CorrectionAction.REPAIR_ENCODING: _apply_mapping,
        CorrectionAction.CAST_TO_NUMERIC: _cast_numeric,
        CorrectionAction.PARSE_DATES: _parse_dates,
        CorrectionAction.CLEAR_INVALID_VALUES: _clear_invalid,
        CorrectionAction.FILL_MISSING: _fill_missing,
        CorrectionAction.CLIP_TO_RANGE: _clip_to_range,
    }
    handler = handlers.get(action)
    if handler is None:
        raise CorrectionError(f"No handler for action '{action.value}'.")

    before = frame[column]
    after = handler(before, proposal)
    changed = int(
        (before.astype("string").fillna("\x00") != after.astype("string").fillna("\x00")).sum()
    )

    result = frame.copy()
    result[column] = after
    return result, changed


# --------------------------------------------------------------------------- handlers


def _drop_duplicates(frame: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    normalised = frame.copy()
    for column in normalised.columns:
        series = normalised[column]
        if series.dtype == object or pd.api.types.is_string_dtype(series):
            normalised[column] = series.astype("string").str.strip().str.casefold()

    keep = ~normalised.duplicated(keep="first")
    removed = int((~keep).sum())
    return frame[keep].reset_index(drop=True), removed


def _fill_from_related(
    frame: pd.DataFrame, proposal: CorrectionProposal
) -> tuple[pd.DataFrame, int]:
    """Fill gaps by looking the value up in the column that determines it.

    Only rows whose key is present *and* known are touched. A gap whose key is itself
    missing stays a gap: this correction recovers recorded values, it does not extend
    the mapping to cases the file never covered.
    """
    column = proposal.column
    determinant = str(proposal.parameters.get("determinant", ""))
    mapping = proposal.parameters.get("mapping")

    if column is None or column not in frame.columns:
        raise CorrectionError(f"Column '{column}' is not present in the dataset.")
    if determinant not in frame.columns:
        raise CorrectionError(f"The determining column '{determinant}' is not in the dataset.")
    if not isinstance(mapping, dict) or not mapping:
        raise CorrectionError("The proposal carries no lookup table.")

    target = to_clean_strings(frame[column])
    keys = to_clean_strings(frame[determinant]).astype("string").str.strip()
    supplied = keys.map(lambda key: mapping.get(key), na_action="ignore").astype("string")

    fillable = missing_mask(frame[column]) & supplied.notna()
    after = target.mask(fillable, other=supplied)

    result = frame.copy()
    result[column] = after
    return result, int(fillable.sum())


def _strip_whitespace(series: pd.Series, proposal: CorrectionProposal) -> pd.Series:  # noqa: ARG001
    return to_clean_strings(series).str.strip()


def _apply_mapping(series: pd.Series, proposal: CorrectionProposal) -> pd.Series:
    mapping = proposal.parameters.get("mapping")
    if not isinstance(mapping, dict) or not mapping:
        raise CorrectionError("The proposal carries no value mapping.")
    text = to_clean_strings(series)
    return text.map(lambda value: mapping.get(value, value), na_action="ignore")


def _cast_numeric(series: pd.Series, proposal: CorrectionProposal) -> pd.Series:  # noqa: ARG001
    return coerce_numeric(series)


def _parse_dates(series: pd.Series, proposal: CorrectionProposal) -> pd.Series:
    output_format = str(proposal.parameters.get("output_format", "%Y-%m-%d"))
    parsed = parse_dates_with_format(series, None)
    return parsed.dt.strftime(output_format)


def _clear_invalid(series: pd.Series, proposal: CorrectionProposal) -> pd.Series:
    predicate = str(proposal.parameters.get("predicate", ""))
    text = to_clean_strings(series)

    if predicate == "invalid_email":
        invalid = text.str.strip().map(
            lambda value: not bool(EMAIL_RE.match(value)), na_action="ignore"
        )
    elif predicate == "invalid_date":
        invalid = parse_dates_with_format(series, None).isna() & ~missing_mask(series)
    elif predicate == "placeholder_value":
        targets = proposal.parameters.get("values")
        if not isinstance(targets, list) or not targets:
            raise CorrectionError("The proposal lists no placeholder values to clear.")
        wanted = {str(value).strip().casefold() for value in targets}
        invalid = text.str.strip().str.casefold().isin(wanted)
    else:
        raise CorrectionError(f"Unknown clear-values predicate '{predicate}'.")

    invalid = invalid.astype("boolean").fillna(False).astype(bool)
    return text.mask(invalid, other=pd.NA)


def _fill_missing(series: pd.Series, proposal: CorrectionProposal) -> pd.Series:
    if "value" not in proposal.parameters:
        raise CorrectionError("The proposal carries no fill value.")
    value = str(proposal.parameters["value"])
    return to_clean_strings(series).mask(missing_mask(series), other=value)


def _clip_to_range(series: pd.Series, proposal: CorrectionProposal) -> pd.Series:
    numeric = coerce_numeric(series)
    minimum = proposal.parameters.get("minimum")
    maximum = proposal.parameters.get("maximum")
    return numeric.clip(
        lower=float(minimum) if minimum is not None else None,
        upper=float(maximum) if maximum is not None else None,
    )
