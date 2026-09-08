"""Run the whole pipeline against a real public dataset.

`tests/test_demo_data_detection.py` measures detection against synthetic data whose
defects this project injected on purpose. That measurement is precise and partly
circular: it can only find what someone thought to break.

This file is the counterweight. `data/public/milano_attivita_storiche.csv` is a genuine
open dataset (CC0, no personal data) that nobody here authored, and every assertion
below describes something that is actually wrong with it. Three of the checks the
project now has exist because this file exposed their absence.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from dqcopilot.corrections.proposer import propose_corrections
from dqcopilot.models import CorrectionAction, IssueType
from dqcopilot.services.analysis import AnalysisResult, analyze_bytes

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "public"
DATASET = DATA_DIR / "milano_attivita_storiche.csv"
SOURCE = DATA_DIR / "source.json"

pytestmark = pytest.mark.skipif(
    not DATASET.is_file(),
    reason="Run scripts/fetch_public_dataset.py to download the public dataset.",
)


@pytest.fixture(scope="module")
def analysis() -> AnalysisResult:
    """The full analysis of the real file, computed once for the module."""
    return analyze_bytes(DATASET.read_bytes(), DATASET.name)


def columns_with(analysis: AnalysisResult, issue: IssueType) -> set[str]:
    """Return the columns reported for one issue type."""
    return {
        finding.column
        for finding in analysis.findings
        if finding.issue_type is issue and finding.column
    }


class TestFixtureIntegrity:
    def test_the_committed_bytes_are_the_bytes_that_were_published(self) -> None:
        """Guard against git, an editor or a merge silently rewriting the fixture.

        The encoding defects live in individual bytes. Anything that "helpfully" fixes
        the file - line-ending normalisation, a re-save as UTF-8 - would delete the
        defects while leaving a file that still looks correct.
        """
        expected = json.loads(SOURCE.read_text(encoding="utf-8"))
        raw = DATASET.read_bytes()

        assert len(raw) == expected["bytes"]
        assert hashlib.sha256(raw).hexdigest() == expected["sha256"]

    def test_the_licence_permits_redistribution(self) -> None:
        # A dataset committed to a public repository needs a licence that allows it.
        expected = json.loads(SOURCE.read_text(encoding="utf-8"))
        assert "CCZero" in expected["licence"]


class TestItReadsTheFile:
    def test_detects_the_semicolon_delimiter(self, analysis: AnalysisResult) -> None:
        # Read with the wrong delimiter this file is one column wide.
        assert analysis.column_count == 19
        assert analysis.row_count == 546

    def test_reports_the_business_rules_it_could_not_evaluate(
        self, analysis: AnalysisResult
    ) -> None:
        """The demo rules reference columns this dataset does not have.

        Skipping them is correct. Saying so is the point: a rule set that silently
        evaluates nothing reports a perfect score.
        """
        skipped = [
            finding
            for finding in analysis.findings
            if finding.issue_type is IssueType.BUSINESS_RULE_VIOLATION
        ]
        assert skipped, "the skipped rules must be reported, not passed over in silence"
        assert "not evaluated" in skipped[0].title


class TestDefectsInRealData:
    def test_finds_the_double_encoded_company_names(self, analysis: AnalysisResult) -> None:
        assert "DENOM_IMPRES" in columns_with(analysis, IssueType.CORRUPTED_ENCODING)

    def test_finds_the_legal_form_used_as_a_company_name(self, analysis: AnalysisResult) -> None:
        finding = next(f for f in analysis.findings if f.issue_type is IssueType.PLACEHOLDER_VALUE)
        assert finding.column == "DENOM_IMPRES"
        assert "ditta individuale" in finding.details["values"]

    def test_finds_the_sentinel_year_and_the_out_of_area_postcode(
        self, analysis: AnalysisResult
    ) -> None:
        outliers = columns_with(analysis, IssueType.SUSPICIOUS_NUMERIC)
        # 9999 stands for "unknown"; 20055 is not a Milan postcode.
        assert {"ANNO_RICONOSCIMENTO", "ZIP"} <= outliers

    def test_finds_the_column_that_says_the_same_thing_on_every_row(
        self, analysis: AnalysisResult
    ) -> None:
        assert "COMUNE" in columns_with(analysis, IssueType.CONSTANT_COLUMN)


@pytest.fixture(scope="module")
def filled_columns(analysis: AnalysisResult) -> set[str]:
    """Columns the tool is willing to fill in automatically."""
    proposals = propose_corrections(analysis.findings.sorted(), analysis.frame, analysis.profile)
    return {
        proposal.column
        for proposal in proposals
        if proposal.action is CorrectionAction.FILL_MISSING and proposal.column
    }


class TestItRefusesToInventLabels:
    @pytest.mark.parametrize("column", ["MUNICIPIO", "ID_NIL", "geo_x", "geo_y"])
    def test_never_offers_to_impute_a_code_or_a_coordinate(
        self, filled_columns: set[str], column: str
    ) -> None:
        """These are the suggestions the tool used to make, and they were wrong.

        The median municipality of a list of shops is a real municipality, and it is not
        this shop's. Being arithmetically valid is what makes the answer dangerous.
        """
        assert column not in filled_columns
