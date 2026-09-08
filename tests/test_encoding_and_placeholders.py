"""Tests for the two defects that only real data taught us to look for.

Both checks were written after running the pipeline against a public dataset from the
Comune di Milano, which scored 99.8/100 while containing double-encoded company names
and a legal form typed into the company-name field. The cases below are the ones that
file actually contained, plus the false positives each check has to avoid.
"""

from __future__ import annotations

import pandas as pd
import pytest

from dqcopilot.corrections.applier import apply_corrections
from dqcopilot.corrections.proposer import propose_corrections
from dqcopilot.encoding_repair import has_lost_characters, repair_mojibake
from dqcopilot.models import CorrectionAction, IssueType, Severity
from dqcopilot.models.corrections import CorrectionDecision
from dqcopilot.naming import looks_like_code, looks_like_coordinate
from dqcopilot.profiling import profile_dataset
from dqcopilot.validation import CheckContext, run_checks


def analyse(frame: pd.DataFrame, only: list[str] | None = None) -> dict[IssueType, list]:
    profile = profile_dataset(frame, "test")
    findings = run_checks(CheckContext(frame=frame, profile=profile), only=only)
    grouped: dict[IssueType, list] = {}
    for finding in findings.findings:
        grouped.setdefault(finding.issue_type, []).append(finding)
    return grouped


class TestMojibakeRepair:
    @pytest.mark.parametrize(
        ("damaged", "expected"),
        [
            # The three that were really in the Milan file.
            ("Garbagnati le SpecialitÃ  srl", "Garbagnati le Specialità srl"),
            ("Cuoccio AntichitÃ  sas", "Cuoccio Antichità sas"),
            (
                "Gestioni AttivitÃ  Commerciali S.r.l.",
                "Gestioni Attività Commerciali S.r.l.",
            ),
            # Other shapes the same bug takes.
            ("CafÃ©", "Café"),
            ("lâ€™azienda", "l’azienda"),
            ("Â£50", "£50"),
        ],
    )
    def test_repairs_double_encoded_text(self, damaged: str, expected: str) -> None:
        assert repair_mojibake(damaged) == expected

    @pytest.mark.parametrize(
        "value",
        [
            "Viganò Alta Moda S.r.l.",  # correctly encoded Italian
            "Café de Paris",
            "Frères Müller",
            "São Paulo",
            "Ångström",
            "Đặng Văn",  # outside latin-1 entirely
            "MILANO",
            "20123",
            "",
        ],
    )
    def test_leaves_correctly_encoded_text_alone(self, value: str) -> None:
        assert repair_mojibake(value) is None

    def test_reports_but_does_not_repair_discarded_bytes(self) -> None:
        lost = "Citt� di Londra"
        assert has_lost_characters(lost)
        assert repair_mojibake(lost) is None


class TestCorruptedEncodingCheck:
    def test_detects_and_never_files_corruption_as_cosmetic(self) -> None:
        frame = pd.DataFrame({"name": ["Rossi srl", "SpecialitÃ  srl", *["Bianchi spa"] * 98]})
        finding = analyse(frame)[IssueType.CORRUPTED_ENCODING][0]

        assert finding.affected_rows == 1
        # One row in a hundred is 1%, which every ratio rule would call low. Silent
        # corruption is not low, so the check floors it at medium.
        assert finding.severity is Severity.MEDIUM

    def test_separates_recoverable_damage_from_lost_bytes(self) -> None:
        frame = pd.DataFrame({"name": ["CafÃ©", "Citt�", "Rossi"]})
        finding = analyse(frame)[IssueType.CORRUPTED_ENCODING][0]

        assert finding.affected_rows == 2
        assert finding.details["unrecoverable"] == 1
        assert "cannot be recovered" in finding.explanation

    def test_stays_quiet_on_a_correctly_encoded_column(self) -> None:
        frame = pd.DataFrame({"name": ["Viganò", "Café", "Müller", "São Paulo"]})
        assert IssueType.CORRUPTED_ENCODING not in analyse(frame)

    def test_repair_proposal_restores_the_original_text(self) -> None:
        frame = pd.DataFrame({"name": ["SpecialitÃ  srl", "Rossi srl"]})
        profile = profile_dataset(frame, "test")
        findings = run_checks(CheckContext(frame=frame, profile=profile))
        proposals = propose_corrections(findings.sorted(), frame, profile)

        repair = next(p for p in proposals if p.action is CorrectionAction.REPAIR_ENCODING)
        decisions = {repair.proposal_id: CorrectionDecision.approve(repair.proposal_id, "test")}
        cleaned, changes = apply_corrections(frame, proposals, decisions)

        assert cleaned["name"].tolist() == ["Specialità srl", "Rossi srl"]
        assert changes[0].rows_changed == 1


class TestPlaceholderValueCheck:
    def test_detects_filler_words(self) -> None:
        frame = pd.DataFrame(
            {"supplier": ["Rossi srl", "N/A", "unknown", "Bianchi spa", "-", "da definire"]}
        )
        finding = analyse(frame)[IssueType.PLACEHOLDER_VALUE][0]
        assert finding.affected_rows == 4

    def test_detects_a_domain_placeholder_no_word_list_would_contain(self) -> None:
        # The Milan case: a legal form typed into a company-name column. It is only
        # detectable because the column is otherwise a list of distinct names.
        names = [f"Impresa {index}" for index in range(60)]
        names[5:15] = ["ditta individuale"] * 10
        frame = pd.DataFrame({"denominazione": names})

        finding = analyse(frame)[IssueType.PLACEHOLDER_VALUE][0]
        assert finding.affected_rows == 10
        assert "ditta individuale" in finding.details["values"]

    def test_groups_a_placeholder_written_in_two_cases(self) -> None:
        names = [f"Impresa {index}" for index in range(60)]
        names[5:12] = ["ditta individuale"] * 4 + ["Ditta Individuale"] * 3
        frame = pd.DataFrame({"denominazione": names})

        assert analyse(frame)[IssueType.PLACEHOLDER_VALUE][0].affected_rows == 7

    def test_does_not_call_a_repeated_category_a_placeholder(self) -> None:
        # 'Hardware' repeats because it is what the column is for, not because it is
        # standing in for something absent.
        frame = pd.DataFrame({"category": ["Hardware", "Software", "Hardware"] * 20})
        assert IssueType.PLACEHOLDER_VALUE not in analyse(frame)

    def test_does_not_report_nd_in_a_column_of_short_codes(self) -> None:
        # "ND" is North Dakota here, not "non disponibile".
        frame = pd.DataFrame({"state": ["CA", "NY", "ND", "TX", "ND", "FL"]})
        assert IssueType.PLACEHOLDER_VALUE not in analyse(frame)

    def test_does_not_report_two_records_that_share_a_name(self) -> None:
        names = [f"Impresa {index}" for index in range(60)]
        names[7] = names[3]
        frame = pd.DataFrame({"denominazione": names})
        assert IssueType.PLACEHOLDER_VALUE not in analyse(frame)

    def test_clearing_makes_the_gap_visible_instead_of_inventing_data(self) -> None:
        frame = pd.DataFrame({"supplier": ["Rossi srl", "N/A", "Bianchi spa", "unknown"]})
        profile = profile_dataset(frame, "test")
        findings = run_checks(CheckContext(frame=frame, profile=profile))
        proposals = propose_corrections(findings.sorted(), frame, profile)

        clear = next(
            p
            for p in proposals
            if p.action is CorrectionAction.CLEAR_INVALID_VALUES
            and p.parameters.get("predicate") == "placeholder_value"
        )
        assert clear.destructive is True

        decisions = {clear.proposal_id: CorrectionDecision.approve(clear.proposal_id, "test")}
        cleaned, _ = apply_corrections(frame, proposals, decisions)

        assert cleaned["supplier"].isna().sum() == 2
        assert cleaned["supplier"].tolist()[0] == "Rossi srl"


class TestImputationRefusesLabels:
    @pytest.mark.parametrize(
        "column",
        ["municipio", "MUNICIPIO", "zip_code", "postcode", "id_nil", "region_code", "provincia"],
    )
    def test_recognises_codes(self, column: str) -> None:
        assert looks_like_code(column) is True

    @pytest.mark.parametrize("column", ["quantity", "revenue", "unit_price", "age", "score"])
    def test_leaves_real_quantities_alone(self, column: str) -> None:
        assert looks_like_code(column) is False
        assert looks_like_coordinate(column) is False

    @pytest.mark.parametrize("column", ["geo_x", "latitude", "lon", "coordinates"])
    def test_recognises_coordinates(self, column: str) -> None:
        assert looks_like_coordinate(column) is True

    def test_offers_the_median_for_a_genuine_measurement(self) -> None:
        frame = pd.DataFrame({"quantity": [1, 2, 3, None, 5, 6, 7, 8, 9, 10]})
        profile = profile_dataset(frame, "test")
        findings = run_checks(CheckContext(frame=frame, profile=profile))
        proposals = propose_corrections(findings.sorted(), frame, profile)

        assert any(p.action is CorrectionAction.FILL_MISSING for p in proposals)

    @pytest.mark.parametrize("column", ["municipio", "geo_x"])
    def test_refuses_to_impute_a_code_or_a_coordinate(self, column: str) -> None:
        frame = pd.DataFrame({column: [1, 2, 3, None, 5, 6, 7, 8, 9, 10]})
        profile = profile_dataset(frame, "test")
        findings = run_checks(CheckContext(frame=frame, profile=profile))
        proposals = propose_corrections(findings.sorted(), frame, profile)

        actions = {proposal.action for proposal in proposals}
        assert CorrectionAction.FILL_MISSING not in actions
        assert CorrectionAction.MANUAL_REVIEW in actions
