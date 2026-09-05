from __future__ import annotations

import pandas as pd
import pytest

from dqcopilot.config import Settings
from dqcopilot.corrections import apply_corrections, propose_corrections
from dqcopilot.export import neutralise_formula, sanitise_for_export, to_csv_bytes, to_xlsx_bytes
from dqcopilot.models import CorrectionAction, DecisionStatus, IssueType
from dqcopilot.models.corrections import CorrectionDecision
from dqcopilot.profiling import profile_dataset
from dqcopilot.services import ReviewSession, analyze_bytes
from dqcopilot.validation import CheckContext, run_checks


def build_session(frame: pd.DataFrame) -> ReviewSession:
    """Analyse a frame in memory and open a review session on it."""
    from dqcopilot.scoring import compute_quality_score
    from dqcopilot.services.analysis import AnalysisResult

    profile = profile_dataset(frame, "test")
    findings = run_checks(CheckContext(frame=frame, profile=profile))
    analysis = AnalysisResult(
        analysis_id="test",
        source_name="test.csv",
        frame=frame,
        profile=profile,
        findings=findings,
        score=compute_quality_score(profile, findings),
        row_count=int(frame.shape[0]),
        column_count=int(frame.shape[1]),
    )
    return ReviewSession.from_analysis(analysis)


def proposals_for(frame: pd.DataFrame, action: CorrectionAction) -> list:
    session = build_session(frame)
    return [p for p in session.proposals if p.action is action]


class TestProposalGeneration:
    def test_whitespace_produces_a_strip_proposal(self) -> None:
        frame = pd.DataFrame({"city": ["Paris", " Lyon", "Nice "]})
        proposals = proposals_for(frame, CorrectionAction.STRIP_WHITESPACE)
        assert len(proposals) == 1
        assert proposals[0].column == "city"
        assert not proposals[0].destructive
        assert len(proposals[0].preview) == 2

    def test_preview_makes_padding_visible(self) -> None:
        frame = pd.DataFrame({"city": ["Paris", " Lyon"]})
        preview = proposals_for(frame, CorrectionAction.STRIP_WHITESPACE)[0].preview
        assert preview[0].before == "[ Lyon]"
        assert preview[0].after == "Lyon"

    def test_capitalisation_maps_to_the_most_frequent_spelling(self) -> None:
        frame = pd.DataFrame({"country": ["FR", "FR", "FR", "fr", "Fr"]})
        proposal = proposals_for(frame, CorrectionAction.NORMALIZE_CASE)[0]
        assert proposal.parameters["mapping"] == {"fr": "FR", "Fr": "FR"}

    def test_capitalisation_refuses_to_guess_on_a_tie(self) -> None:
        """With no majority spelling there is no non-arbitrary winner."""
        frame = pd.DataFrame({"country": ["FR", "fr", "DE", "DE"]})
        assert proposals_for(frame, CorrectionAction.NORMALIZE_CASE) == []

    def test_duplicates_produce_a_destructive_proposal(self) -> None:
        frame = pd.DataFrame({"a": ["x", "y", "x"], "b": ["1", "2", "1"]})
        proposal = proposals_for(frame, CorrectionAction.DROP_DUPLICATE_ROWS)[0]
        assert proposal.destructive
        assert proposal.requires_extra_care

    def test_numeric_cast_warns_about_unparseable_values(self) -> None:
        frame = pd.DataFrame({"revenue": ["1,234.56", "€900", "twelve", "77"]})
        proposal = proposals_for(frame, CorrectionAction.CAST_TO_NUMERIC)[0]
        assert proposal.destructive
        assert proposal.parameters["unparseable_becomes_empty"] == 1
        assert "cannot be parsed" in proposal.description

    def test_missing_values_offer_imputation_with_a_warning(self) -> None:
        frame = pd.DataFrame({"country": ["FR", "FR", "FR", "DE", None]})
        proposal = proposals_for(frame, CorrectionAction.FILL_MISSING)[0]
        assert proposal.parameters["value"] == "FR"
        assert "imputation" in proposal.description.lower()
        assert proposal.confidence < 1.0

    def test_fully_empty_column_gets_manual_review_not_imputation(self) -> None:
        frame = pd.DataFrame({"a": [None, None, None], "b": ["x", "y", "z"]})
        session = build_session(frame)
        actions = {p.action for p in session.proposals if p.column == "a"}
        assert CorrectionAction.FILL_MISSING not in actions
        assert CorrectionAction.MANUAL_REVIEW in actions

    def test_probable_duplicates_are_never_auto_merged(self) -> None:
        frame = pd.DataFrame(
            {
                "supplier_name": ["Acme Ltd", "ACME Limited.", "Globex GmbH", "Initech"],
                "city": ["Paris", "Lyon", "Berlin", "Austin"],
            }
        )
        session = build_session(frame)
        merge_proposals = [
            p
            for p in session.proposals
            if p.finding_id
            in {
                f.finding_id
                for f in session.analysis.findings.findings
                if f.issue_type is IssueType.PROBABLE_DUPLICATE_ROWS
            }
        ]
        assert merge_proposals
        assert all(p.action is CorrectionAction.MANUAL_REVIEW for p in merge_proposals)

    def test_outliers_are_flagged_for_review_only(self) -> None:
        values = [str(100 + i) for i in range(50)] + ["99999999"]
        frame = pd.DataFrame({"measurement": values})
        session = build_session(frame)
        outlier_ids = {
            f.finding_id
            for f in session.analysis.findings.findings
            if f.issue_type is IssueType.SUSPICIOUS_NUMERIC
        }
        related = [p for p in session.proposals if p.finding_id in outlier_ids]
        assert related and all(p.action is CorrectionAction.MANUAL_REVIEW for p in related)


class TestApprovalWorkflow:
    def test_nothing_is_applied_without_approval(self) -> None:
        frame = pd.DataFrame({"city": ["Paris", " Lyon", "Nice "]})
        session = build_session(frame)
        result = session.build_cleaned_dataset()

        assert result.applied == []
        pd.testing.assert_frame_equal(result.frame, frame)

    def test_rejecting_everything_changes_nothing(self) -> None:
        frame = pd.DataFrame({"city": ["Paris", " Lyon", "Nice "]})
        session = build_session(frame)
        for proposal in session.proposals:
            session.reject(proposal.proposal_id)

        result = session.build_cleaned_dataset()
        assert result.applied == []
        assert session.decision_counts()["rejected"] == len(session.proposals)

    def test_approving_one_proposal_applies_only_that_one(self) -> None:
        frame = pd.DataFrame({"city": ["Paris", " Lyon"], "country": ["FR", "fr"]})
        session = build_session(frame)
        strip = next(p for p in session.proposals if p.action is CorrectionAction.STRIP_WHITESPACE)
        session.approve(strip.proposal_id)

        result = session.build_cleaned_dataset()
        assert [change.action for change in result.applied] == [CorrectionAction.STRIP_WHITESPACE]
        assert result.frame["city"].tolist() == ["Paris", "Lyon"]
        assert result.frame["country"].tolist() == ["FR", "fr"]  # untouched

    def test_the_original_frame_is_never_mutated(self) -> None:
        frame = pd.DataFrame({"city": ["Paris", " Lyon"]})
        original = frame.copy()
        session = build_session(frame)
        for proposal in session.proposals:
            session.approve(proposal.proposal_id)
        session.build_cleaned_dataset()

        pd.testing.assert_frame_equal(frame, original)

    def test_decisions_are_recorded_with_a_timestamp_and_reviewer(self) -> None:
        frame = pd.DataFrame({"city": ["Paris", " Lyon"]})
        session = build_session(frame)
        proposal = session.proposals[0]
        session.approve(proposal.proposal_id)

        decision = session.decisions[proposal.proposal_id]
        assert decision.status is DecisionStatus.APPROVED
        assert decision.decided_at is not None
        assert decision.decided_by == "demo-user"

    def test_a_decision_can_be_changed(self) -> None:
        frame = pd.DataFrame({"city": ["Paris", " Lyon"]})
        session = build_session(frame)
        proposal_id = session.proposals[0].proposal_id

        session.approve(proposal_id)
        assert session.status(proposal_id) is DecisionStatus.APPROVED
        session.reject(proposal_id)
        assert session.status(proposal_id) is DecisionStatus.REJECTED

    def test_unknown_proposal_id_is_rejected(self) -> None:
        session = build_session(pd.DataFrame({"a": ["x"]}))
        with pytest.raises(KeyError):
            session.approve("not-a-real-id")

    def test_manual_review_items_are_never_applied(self) -> None:
        values = [str(100 + i) for i in range(50)] + ["99999999"]
        frame = pd.DataFrame({"measurement": values})
        session = build_session(frame)
        for proposal in session.proposals:
            session.approve(proposal.proposal_id)

        result = session.build_cleaned_dataset()
        assert all(change.action is not CorrectionAction.MANUAL_REVIEW for change in result.applied)


class TestApplyingCorrections:
    def test_strip_whitespace(self) -> None:
        frame = pd.DataFrame({"city": ["Paris", " Lyon", "Nice "]})
        session = build_session(frame)
        session.approve(
            next(
                p for p in session.proposals if p.action is CorrectionAction.STRIP_WHITESPACE
            ).proposal_id
        )
        result = session.build_cleaned_dataset()
        assert result.frame["city"].tolist() == ["Paris", "Lyon", "Nice"]
        assert result.applied[0].rows_changed == 2

    def test_normalise_case(self) -> None:
        frame = pd.DataFrame({"country": ["FR", "FR", "FR", "fr"]})
        session = build_session(frame)
        session.approve(
            next(
                p for p in session.proposals if p.action is CorrectionAction.NORMALIZE_CASE
            ).proposal_id
        )
        result = session.build_cleaned_dataset()
        assert result.frame["country"].tolist() == ["FR"] * 4

    def test_cast_to_numeric(self) -> None:
        frame = pd.DataFrame({"revenue": ["1,234.56", "€900", "(500)", "77"]})
        session = build_session(frame)
        session.approve(
            next(
                p for p in session.proposals if p.action is CorrectionAction.CAST_TO_NUMERIC
            ).proposal_id
        )
        result = session.build_cleaned_dataset()
        assert result.frame["revenue"].tolist() == [1234.56, 900.0, -500.0, 77.0]

    def test_drop_duplicates_removes_rows(self) -> None:
        frame = pd.DataFrame({"a": ["x", "y", "x"], "b": ["1", "2", "1"]})
        session = build_session(frame)
        session.approve(
            next(
                p for p in session.proposals if p.action is CorrectionAction.DROP_DUPLICATE_ROWS
            ).proposal_id
        )
        result = session.build_cleaned_dataset()
        assert result.rows_after == 2
        assert result.rows_removed == 1

    def test_clear_invalid_emails(self) -> None:
        frame = pd.DataFrame({"email": ["a@example.com", "broken@", "c@example.com"]})
        session = build_session(frame)
        session.approve(
            next(
                p for p in session.proposals if p.action is CorrectionAction.CLEAR_INVALID_VALUES
            ).proposal_id
        )
        result = session.build_cleaned_dataset()
        assert pd.isna(result.frame["email"].iloc[1])
        assert result.frame["email"].iloc[0] == "a@example.com"

    def test_corrections_are_applied_in_a_fixed_order(self) -> None:
        """Whitespace is normalised before de-duplication, so padded copies collapse."""
        frame = pd.DataFrame({"a": ["x", " x ", "y"], "b": ["1", "1", "2"]})
        session = build_session(frame)
        for proposal in session.proposals:
            if proposal.action in (
                CorrectionAction.STRIP_WHITESPACE,
                CorrectionAction.DROP_DUPLICATE_ROWS,
            ):
                session.approve(proposal.proposal_id)

        result = session.build_cleaned_dataset()
        assert result.rows_after == 2

    def test_score_improves_after_cleaning(self) -> None:
        frame = pd.DataFrame(
            {
                "city": ["Paris", " Lyon", "Nice ", "Metz"],
                "country": ["FR", "FR", "FR", "fr"],
            }
        )
        session = build_session(frame)
        for proposal in session.actionable:
            session.approve(proposal.proposal_id)

        result = session.build_cleaned_dataset()
        assert result.score_after.overall > result.score_before.overall
        assert result.score_delta > 0


class TestExport:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            ("=1+1", "'=1+1"),
            ("+SUM(A1)", "'+SUM(A1)"),
            ("@import", "'@import"),
            ("=cmd|'/c calc'!A1", "'=cmd|'/c calc'!A1"),
            ("-42.5", "-42.5"),  # a negative number is data, not a formula
            ("-not a number", "'-not a number"),
            ("Paris", "Paris"),
            ("", ""),
        ],
    )
    def test_neutralises_formula_payloads(self, value: str, expected: str) -> None:
        assert neutralise_formula(value) == expected

    def test_sanitises_the_whole_frame(self) -> None:
        frame = pd.DataFrame({"a": ["=1+1", "safe"], "b": [1, 2]})
        safe = sanitise_for_export(frame)
        assert safe["a"].tolist() == ["'=1+1", "safe"]
        assert safe["b"].tolist() == [1, 2]

    def test_csv_export_round_trips(self) -> None:
        frame = pd.DataFrame({"a": ["x", "y"], "b": ["1", "2"]})
        payload = to_csv_bytes(frame)
        assert payload.startswith(b"\xef\xbb\xbf")  # BOM so Excel reads UTF-8
        assert b"a,b" in payload

    def test_xlsx_export_is_a_valid_workbook(self) -> None:
        frame = pd.DataFrame({"a": ["x", "y"]})
        payload = to_xlsx_bytes(frame)
        assert payload.startswith(b"PK\x03\x04")

    def test_exported_csv_can_be_read_back(self, settings: Settings) -> None:
        frame = pd.DataFrame({"city": ["Paris", "Lyon"], "count": ["1", "2"]})
        payload = to_csv_bytes(frame)
        result = analyze_bytes(payload, "cleaned.csv", settings=settings)
        assert result.row_count == 2
        assert list(result.frame.columns) == ["city", "count"]


class TestApplierDirectly:
    def test_only_approved_decisions_are_applied(self) -> None:
        frame = pd.DataFrame({"city": ["Paris", " Lyon"]})
        profile = profile_dataset(frame, "t")
        findings = run_checks(CheckContext(frame=frame, profile=profile))
        proposals = propose_corrections(findings.sorted(), frame, profile)

        strip = next(p for p in proposals if p.action is CorrectionAction.STRIP_WHITESPACE)
        decisions = {strip.proposal_id: CorrectionDecision.reject(strip.proposal_id)}

        cleaned, applied = apply_corrections(frame, proposals, decisions)
        assert applied == []
        assert cleaned["city"].tolist() == ["Paris", " Lyon"]

    def test_a_correction_referencing_a_missing_column_is_skipped(self) -> None:
        frame = pd.DataFrame({"city": ["Paris", " Lyon"]})
        profile = profile_dataset(frame, "t")
        findings = run_checks(CheckContext(frame=frame, profile=profile))
        proposals = propose_corrections(findings.sorted(), frame, profile)

        strip = next(p for p in proposals if p.action is CorrectionAction.STRIP_WHITESPACE)
        strip.column = "does_not_exist"
        decisions = {strip.proposal_id: CorrectionDecision.approve(strip.proposal_id)}

        cleaned, applied = apply_corrections(frame, proposals, decisions)
        assert applied == []
        pd.testing.assert_frame_equal(cleaned, frame)
