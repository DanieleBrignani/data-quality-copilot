"""Tests for recovering a value from the column that decides it.

This closes the last gap the Milan dataset exposed. `NIL` (a neighbourhood name) was
being offered "fill with the most frequent value", which is wrong in the same way the
median municipality was wrong: it is a place, not a quantity. Its value is decided by
`ID_NIL`, so a gap in it has a correct answer to look up - or, when the determinant is
missing too, no answer this file can give.
"""

from __future__ import annotations

import pandas as pd
import pytest

from dqcopilot.corrections.applier import apply_corrections
from dqcopilot.corrections.proposer import propose_corrections
from dqcopilot.models import CorrectionAction
from dqcopilot.models.corrections import CorrectionDecision
from dqcopilot.profiling import profile_dataset
from dqcopilot.profiling.dependencies import build_lookup, find_determinant
from dqcopilot.validation import CheckContext, run_checks


def countries(gaps: list[int], missing_keys: list[int] | None = None) -> pd.DataFrame:
    """Thirty rows where the country code decides the country name."""
    codes = ["IT"] * 10 + ["FR"] * 10 + ["DE"] * 10
    names = ["Italia"] * 10 + ["Francia"] * 10 + ["Germania"] * 10
    for row in gaps:
        names[row] = None
    for row in missing_keys or []:
        codes[row] = None
    return pd.DataFrame({"country_code": codes, "country_name": names})


def proposals_for(frame: pd.DataFrame) -> list:
    profile = profile_dataset(frame, "test")
    findings = run_checks(CheckContext(frame=frame, profile=profile))
    return propose_corrections(findings.sorted(), frame, profile)


class TestFindingTheDeterminant:
    def test_finds_the_column_that_decides_the_value(self) -> None:
        dependency = find_determinant(countries(gaps=[0, 10]), "country_name")

        assert dependency is not None
        assert dependency.determinant == "country_code"
        assert dependency.distinct_keys == 3
        assert dependency.recoverable_rows == 2
        assert dependency.unrecoverable_rows == 0

    def test_counts_a_gap_as_unrecoverable_when_its_key_is_missing_too(self) -> None:
        dependency = find_determinant(countries(gaps=[0, 20], missing_keys=[20]), "country_name")

        assert dependency is not None
        assert dependency.recoverable_rows == 1
        assert dependency.unrecoverable_rows == 1

    def test_does_not_promise_a_fix_for_a_key_never_seen_beside_a_value(self) -> None:
        """A key that only ever appears on empty rows teaches the lookup nothing.

        Counting it would advertise a correction the applier could not deliver, and the
        audit log would then disagree with the proposal.
        """
        frame = pd.DataFrame(
            {
                "country_code": ["IT"] * 10 + ["FR"] * 10 + ["ES"] * 5,
                "country_name": ["Italia"] * 10 + ["Francia"] * 10 + [None] * 5,
            }
        )
        dependency = find_determinant(frame, "country_name")

        assert dependency is not None
        assert dependency.recoverable_rows == 0
        assert dependency.unrecoverable_rows == 5

    def test_a_primary_key_is_not_a_determinant(self) -> None:
        # A unique id decides every column by construction. Saying so is true and useless.
        frame = countries(gaps=[0])
        frame.insert(0, "row_id", [f"R{index:03d}" for index in range(len(frame))])

        dependency = find_determinant(frame, "country_name")
        assert dependency is not None
        assert dependency.determinant == "country_code"

    def test_ignores_agreement_across_too_few_rows(self) -> None:
        frame = pd.DataFrame({"a": ["x", "y", None], "b": ["1", "2", "3"]})
        assert find_determinant(frame, "a") is None

    def test_returns_nothing_when_no_column_decides(self) -> None:
        frame = pd.DataFrame(
            {
                "amount": [float(index) for index in range(30)],
                "note": [f"note {index}" for index in range(29)] + [None],
            }
        )
        assert find_determinant(frame, "amount") is None

    def test_the_lookup_contains_only_observed_pairs(self) -> None:
        frame = countries(gaps=[0, 20], missing_keys=[20])
        dependency = find_determinant(frame, "country_name")
        assert dependency is not None

        assert build_lookup(frame, dependency) == {
            "IT": "Italia",
            "FR": "Francia",
            "DE": "Germania",
        }


class TestRecoveryBeatsImputation:
    def test_recovers_the_gaps_whose_key_is_present(self) -> None:
        frame = countries(gaps=[0, 10, 20], missing_keys=[20])
        proposals = proposals_for(frame)

        recover = next(p for p in proposals if p.action is CorrectionAction.FILL_FROM_RELATED)
        assert recover.column == "country_name"
        assert recover.affected_rows == 2
        assert recover.destructive is False

        decisions = {recover.proposal_id: CorrectionDecision.approve(recover.proposal_id, "t")}
        cleaned, changes = apply_corrections(frame, proposals, decisions)

        assert cleaned["country_name"][0] == "Italia"
        assert cleaned["country_name"][10] == "Francia"
        # The key is missing on this row, so the gap survives on purpose.
        assert pd.isna(cleaned["country_name"][20])
        assert changes[0].rows_changed == 2

    def test_the_applied_count_matches_what_the_proposal_promised(self) -> None:
        frame = countries(gaps=[0, 10, 20], missing_keys=[20])
        proposals = proposals_for(frame)
        recover = next(p for p in proposals if p.action is CorrectionAction.FILL_FROM_RELATED)

        decisions = {recover.proposal_id: CorrectionDecision.approve(recover.proposal_id, "t")}
        _, changes = apply_corrections(frame, proposals, decisions)

        assert changes[0].rows_changed == recover.affected_rows

    def test_explains_itself_instead_of_averaging_when_nothing_can_be_recovered(self) -> None:
        frame = countries(gaps=[0, 10, 20], missing_keys=[0, 10, 20])
        proposals = proposals_for(frame)

        name = next(p for p in proposals if p.column == "country_name")
        assert name.action is CorrectionAction.MANUAL_REVIEW
        assert "decided by 'country_code'" in name.description
        assert "has to come from the source system" in name.description

    def test_a_genuine_measurement_still_gets_the_median(self) -> None:
        # The guard must not swallow ordinary imputation.
        frame = pd.DataFrame({"quantity": [float(index) for index in range(29)] + [None]})
        proposals = proposals_for(frame)

        assert any(p.action is CorrectionAction.FILL_MISSING for p in proposals)


class TestApplierGuards:
    @pytest.mark.parametrize(
        "parameters",
        [
            {"determinant": "missing_column", "mapping": {"IT": "Italia"}},
            {"determinant": "country_code", "mapping": {}},
            {"determinant": "country_code"},
        ],
    )
    def test_a_malformed_proposal_changes_nothing(self, parameters: dict) -> None:
        """A broken correction is skipped; the other approved ones still apply."""
        frame = countries(gaps=[0])
        proposals = proposals_for(frame)
        recover = next(p for p in proposals if p.action is CorrectionAction.FILL_FROM_RELATED)
        broken = recover.model_copy(update={"parameters": parameters})

        decisions = {broken.proposal_id: CorrectionDecision.approve(broken.proposal_id, "t")}
        cleaned, changes = apply_corrections(frame, [broken], decisions)

        assert changes == []
        assert pd.isna(cleaned["country_name"][0])
