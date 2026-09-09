from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dqcopilot.models import IssueType, Severity
from dqcopilot.profiling import profile_dataset
from dqcopilot.validation import CheckContext, run_checks
from dqcopilot.validation.registry import registered_checks


def analyse(frame: pd.DataFrame) -> dict[IssueType, list]:
    """Run every registered check and group findings by issue type."""
    profile = profile_dataset(frame, "test")
    findings = run_checks(CheckContext(frame=frame, profile=profile))
    grouped: dict[IssueType, list] = {}
    for finding in findings.findings:
        grouped.setdefault(finding.issue_type, []).append(finding)
    return grouped


class TestMissingValues:
    def test_detects_missing_values(self) -> None:
        frame = pd.DataFrame({"a": ["x", None, "  ", "y"], "b": ["1", "2", "3", "4"]})
        found = analyse(frame)[IssueType.MISSING_VALUES]
        assert len(found) == 1
        assert found[0].column == "a"
        assert found[0].affected_rows == 2
        assert found[0].row_indices == [1, 2]

    def test_no_finding_on_complete_data(self) -> None:
        frame = pd.DataFrame({"a": ["x", "y"], "b": ["1", "2"]})
        assert IssueType.MISSING_VALUES not in analyse(frame)

    def test_fully_empty_column_is_critical(self) -> None:
        frame = pd.DataFrame({"a": [None, None, None], "b": ["1", "2", "3"]})
        found = analyse(frame)[IssueType.MISSING_VALUES]
        assert found[0].severity is Severity.CRITICAL

    def test_severity_scales_with_ratio(self) -> None:
        mostly_present = pd.DataFrame({"a": ["x"] * 99 + [None]})
        mostly_absent = pd.DataFrame({"a": ["x"] * 20 + [None] * 80})
        assert analyse(mostly_present)[IssueType.MISSING_VALUES][0].severity is Severity.LOW
        assert analyse(mostly_absent)[IssueType.MISSING_VALUES][0].severity is Severity.CRITICAL


class TestExactDuplicates:
    def test_detects_exact_duplicates(self) -> None:
        frame = pd.DataFrame({"a": ["x", "y", "x"], "b": ["1", "2", "1"]})
        found = analyse(frame)[IssueType.EXACT_DUPLICATE_ROWS]
        assert found[0].affected_rows == 1
        assert found[0].row_indices == [2]

    def test_ignores_case_and_padding(self) -> None:
        frame = pd.DataFrame({"a": ["Paris", " paris "], "b": ["1", "1"]})
        assert IssueType.EXACT_DUPLICATE_ROWS in analyse(frame)

    def test_no_finding_when_unique(self) -> None:
        frame = pd.DataFrame({"a": ["x", "y", "z"]})
        assert IssueType.EXACT_DUPLICATE_ROWS not in analyse(frame)


class TestProbableDuplicates:
    def test_detects_company_name_variants(self) -> None:
        frame = pd.DataFrame(
            {
                "supplier_name": ["Acme Ltd", "ACME Limited.", "Globex GmbH", "Initech"],
                "city": ["Paris", "Lyon", "Berlin", "Austin"],
            }
        )
        found = analyse(frame)[IssueType.PROBABLE_DUPLICATE_ROWS]
        assert found[0].affected_rows == 2
        assert found[0].details["duplicate_groups"] == 1

    def test_does_not_double_report_exact_duplicates(self) -> None:
        frame = pd.DataFrame({"customer_name": ["Acme", "Acme"], "city": ["Paris", "Paris"]})
        grouped = analyse(frame)
        assert IssueType.EXACT_DUPLICATE_ROWS in grouped
        assert IssueType.PROBABLE_DUPLICATE_ROWS not in grouped

    def test_no_key_columns_means_no_finding(self) -> None:
        frame = pd.DataFrame({"alpha": ["a", "b"], "beta": ["c", "d"]})
        assert IssueType.PROBABLE_DUPLICATE_ROWS not in analyse(frame)


class TestWhitespace:
    def test_detects_padding(self) -> None:
        frame = pd.DataFrame({"city": ["Paris", " Lyon", "Nice ", "Metz"]})
        found = analyse(frame)[IssueType.LEADING_TRAILING_WHITESPACE]
        assert found[0].affected_rows == 2
        assert found[0].row_indices == [1, 2]

    def test_blank_strings_are_missing_not_padded(self) -> None:
        frame = pd.DataFrame({"city": ["Paris", "   ", "Lyon"]})
        grouped = analyse(frame)
        assert IssueType.LEADING_TRAILING_WHITESPACE not in grouped
        assert IssueType.MISSING_VALUES in grouped


class TestCapitalization:
    def test_detects_case_variants(self) -> None:
        frame = pd.DataFrame({"country": ["FR", "fr", "Fr", "DE", "DE"]})
        found = analyse(frame)[IssueType.INCONSISTENT_CAPITALIZATION]
        assert found[0].affected_rows == 3
        assert set(found[0].details["variant_groups"]["fr"]) == {"FR", "fr", "Fr"}

    def test_consistent_column_is_clean(self) -> None:
        frame = pd.DataFrame({"country": ["FR", "FR", "DE"]})
        assert IssueType.INCONSISTENT_CAPITALIZATION not in analyse(frame)


class TestConstantColumn:
    def test_detects_constant_column(self) -> None:
        frame = pd.DataFrame({"a": ["X"] * 5, "b": list("abcde")})
        found = analyse(frame)[IssueType.CONSTANT_COLUMN]
        assert found[0].column == "a"
        assert found[0].severity is Severity.INFO


class TestCheckRobustness:
    def test_single_row_frame_does_not_crash(self) -> None:
        assert isinstance(analyse(pd.DataFrame({"a": ["x"]})), dict)

    @pytest.mark.parametrize(
        "frame",
        [
            pd.DataFrame({"a": [None, None]}),
            pd.DataFrame({"a": [1, 2, 3], "b": [1.5, 2.5, 3.5]}),
            pd.DataFrame({"d": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"])}),
        ],
    )
    def test_edge_case_frames(self, frame: pd.DataFrame) -> None:
        assert isinstance(analyse(frame), dict)


class TestTheCatalogueMatchesTheDocumentation:
    """The README describes the check suite to someone deciding whether to trust it.

    It has already drifted once - the prose claimed fifteen checks while seventeen were
    registered - and a reader has no way to notice. These two tests make the drift fail
    the build instead.
    """

    def test_every_registered_check_is_documented(self) -> None:
        readme = (Path(__file__).resolve().parents[1] / "README.md").read_text(encoding="utf-8")
        undocumented = [
            check.check_id for check in registered_checks() if f"`{check.check_id}`" not in readme
        ]
        assert not undocumented, f"missing from the README check table: {undocumented}"

    def test_the_stated_count_is_the_real_count(self) -> None:
        readme = (Path(__file__).resolve().parents[1] / "README.md").read_text(encoding="utf-8")
        stated = f"{len(registered_checks())} deterministic checks"
        assert stated in readme, f"the README should say '{stated}'"


class TestTheNormalisedViewIsBuiltOnce:
    """Both duplicate checks compare the same trimmed, case-folded view of the data.

    Each of them used to build it from scratch - two full copies of the dataset to
    answer two questions about one normalisation. Timing that is unreliable on a busy
    machine, so the guarantee is asserted by counting the work instead of clocking it.
    """

    def test_repeated_calls_return_the_same_object(self) -> None:
        frame = pd.DataFrame({"name": [" Alice ", "ALICE", "Bob"], "n": [1, 2, 3]})
        context = CheckContext(frame=frame, profile=profile_dataset(frame, "test"))

        assert context.normalised_frame() is context.normalised_frame()

    def test_columns_without_text_are_shared_rather_than_copied(self) -> None:
        frame = pd.DataFrame({"name": [" Alice ", "ALICE"], "amount": [1.5, 2.5]})
        context = CheckContext(frame=frame, profile=profile_dataset(frame, "test"))
        normalised = context.normalised_frame()

        # Trimming does not apply to a float column, so duplicating it would be cost
        # with no effect. Identity of the array object proves nothing here - pandas
        # hands back a fresh wrapper either way - so the shared buffer is the test.
        assert np.shares_memory(normalised["amount"].to_numpy(), frame["amount"].to_numpy())
        assert normalised["name"].tolist() == ["alice", "alice"]

    def test_normalising_never_alters_the_original(self) -> None:
        frame = pd.DataFrame({"name": [" Alice ", "ALICE"]})
        context = CheckContext(frame=frame, profile=profile_dataset(frame, "test"))
        context.normalised_frame()

        assert frame["name"].tolist() == [" Alice ", "ALICE"]
