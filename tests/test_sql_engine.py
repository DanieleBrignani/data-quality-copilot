"""Tests for the SQL engine, including where it disagrees with the Python one.

A second implementation of the same checks is only worth having if the two are held to
each other. The class at the bottom does that on the demo dataset and pins the two places
they differ, so a *new* difference fails the build while the known ones stay documented
rather than forgotten.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dqcopilot.config import get_settings
from dqcopilot.models import IssueType
from dqcopilot.rules.loader import load_rules_or_none
from dqcopilot.services.analysis import analyze_bytes
from dqcopilot.sql.checks import run_sql_checks
from dqcopilot.sql.engine import SqlEngineError, open_source, quote_identifier

DEMO = Path(__file__).resolve().parents[1] / "data" / "demo" / "customers.csv"

#: Checks both engines implement. Anything outside this set is Python-only by design.
SHARED_CHECKS = frozenset(
    {"missing_values", "constant_column", "exact_duplicate_rows", "business_rule"}
)


def counts_by_finding(findings) -> dict[tuple[str, str | None, str], int]:  # noqa: ANN001
    """Reduce a finding set to {(check, column, title): affected rows} for comparison."""
    return {
        (finding.check_id, finding.column, finding.title): finding.affected_rows
        for finding in findings
        if finding.check_id in SHARED_CHECKS and "not evaluated" not in finding.title
    }


class TestIdentifiersAreSafe:
    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("plain", '"plain"'),
            ("with space", '"with space"'),
            ('evil" ; DROP TABLE t; --', '"evil"" ; DROP TABLE t; --"'),
            ('""', '""""""'),
        ],
    )
    def test_quoting_closes_the_injection(self, name: str, expected: str) -> None:
        assert quote_identifier(name) == expected

    def test_a_hostile_header_cannot_run_a_second_statement(self, tmp_path: Path) -> None:
        """A column name is untrusted input: it comes from a file someone uploaded."""
        hostile = tmp_path / "hostile.csv"
        hostile.write_text(
            'name,"evil"" ; DROP TABLE source; --"\nAlice,1\nBob,\n', encoding="utf-8"
        )

        with open_source(hostile) as source:
            assert 'evil" ; DROP TABLE source; --' in source.columns
            findings = run_sql_checks(source)

        # The query ran, the view survived, and the hostile column was analysed normally.
        assert any(finding.issue_type is IssueType.MISSING_VALUES for finding in findings)


class TestOpeningSources:
    def test_rejects_a_format_it_cannot_read(self, tmp_path: Path) -> None:
        path = tmp_path / "book.xlsx"
        path.write_bytes(b"not really a workbook")
        with pytest.raises(SqlEngineError, match="reads"):
            open_source(path)

    def test_rejects_a_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(SqlEngineError, match="No such file"):
            open_source(tmp_path / "absent.csv")

    def test_reads_shape_without_loading_rows(self) -> None:
        with open_source(DEMO) as source:
            assert source.row_count == 220
            assert len(source.columns) == 9


class TestStructuralChecks:
    def test_counts_empty_and_whitespace_only_cells(self, tmp_path: Path) -> None:
        path = tmp_path / "gaps.csv"
        path.write_text("a,b\n1,x\n,y\n  ,z\n3,\n", encoding="utf-8")

        with open_source(path) as source:
            findings = run_sql_checks(source)

        by_column = {
            finding.column: finding.affected_rows
            for finding in findings
            if finding.issue_type is IssueType.MISSING_VALUES
        }
        assert by_column == {"a": 2, "b": 1}

    def test_reports_a_constant_column(self, tmp_path: Path) -> None:
        path = tmp_path / "constant.csv"
        path.write_text("city,n\nMILANO,1\nMILANO,2\nMILANO,3\n", encoding="utf-8")

        with open_source(path) as source:
            findings = run_sql_checks(source)

        constant = [f for f in findings if f.issue_type is IssueType.CONSTANT_COLUMN]
        assert [f.column for f in constant] == ["city"]

    def test_counts_repeats_not_whole_groups(self, tmp_path: Path) -> None:
        """Three identical rows are two duplicates: the first one is the record."""
        path = tmp_path / "dupes.csv"
        path.write_text("a,b\nx,1\nx,1\nX, 1 \ny,2\n", encoding="utf-8")

        with open_source(path) as source:
            findings = run_sql_checks(source)

        duplicates = next(f for f in findings if f.issue_type is IssueType.EXACT_DUPLICATE_ROWS)
        assert duplicates.affected_rows == 2
        assert duplicates.details["duplicate_groups"] == 1


class TestBusinessRules:
    def test_says_which_rules_it_could_not_evaluate(self) -> None:
        """A rule the engine cannot express must be named, never silently passed."""
        rules = load_rules_or_none(get_settings().rules_file)
        assert rules is not None

        with open_source(DEMO) as source:
            findings = run_sql_checks(source, rules)

        skipped = next(f for f in findings if "not evaluated" in f.title)
        # Date rules need the seventeen-format parser, which SQL does not have.
        assert "signup_date_not_in_future" in skipped.explanation

    def test_a_rule_naming_an_absent_column_is_skipped_not_failed(self, tmp_path: Path) -> None:
        path = tmp_path / "other.csv"
        path.write_text("x\n1\n2\n", encoding="utf-8")
        rules = load_rules_or_none(get_settings().rules_file)

        with open_source(path) as source:
            findings = run_sql_checks(source, rules)

        assert any("not evaluated" in finding.title for finding in findings)
        assert not any(finding.severity.value in {"critical", "high"} for finding in findings)


@pytest.fixture(scope="module")
def comparison() -> tuple[dict, dict]:
    """Both engines' counts for the demo dataset, computed once."""
    rules = load_rules_or_none(get_settings().rules_file)
    python_findings = analyze_bytes(DEMO.read_bytes(), DEMO.name).findings
    with open_source(DEMO) as source:
        sql_findings = run_sql_checks(source, rules)
    return counts_by_finding(python_findings), counts_by_finding(sql_findings)


class TestBothEnginesAgree:
    """The two engines must return the same numbers, or explain why not."""

    #: Where the engines legitimately differ, and why. Anything not listed here must match.
    #:
    #: This set used to have three entries. Two of them described the CSV reader turning
    #: the word "unknown" into an empty cell before Python could see it, which the SQL
    #: engine had no equivalent for. The reader no longer rewrites what a file contains,
    #: and both disappeared - the engines agreeing was independent evidence that the
    #: quieter reader was the more consistent one.
    KNOWN_DIFFERENCES = {
        # Date parsing spans seventeen formats in Python; SQL skips the rule instead of
        # guessing, and says so in its "not evaluated" finding.
        "Business rule failed: signup_date_not_in_future",
    }

    def test_every_shared_finding_reports_the_same_count(
        self, comparison: tuple[dict, dict]
    ) -> None:
        python_counts, sql_counts = comparison

        mismatches = {
            key: (python_counts.get(key), sql_counts.get(key))
            for key in set(python_counts) | set(sql_counts)
            if python_counts.get(key) != sql_counts.get(key)
            and key[2] not in self.KNOWN_DIFFERENCES
        }
        assert not mismatches, f"engines disagree on {mismatches}"

    def test_the_known_differences_are_still_there(self, comparison: tuple[dict, dict]) -> None:
        """If one of these starts matching, the exemption above is stale and should go."""
        python_counts, sql_counts = comparison
        still_differing = {
            key[2]
            for key in set(python_counts) | set(sql_counts)
            if python_counts.get(key) != sql_counts.get(key)
        }
        assert still_differing == self.KNOWN_DIFFERENCES
