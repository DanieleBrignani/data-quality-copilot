from __future__ import annotations

import pytest

from dqcopilot.config import Settings
from dqcopilot.ingestion import UnsupportedFileTypeError
from dqcopilot.models import IssueType
from dqcopilot.services import analyze_bytes


@pytest.mark.e2e
class TestAnalyzeBytes:
    def test_happy_path_csv(self, csv_bytes: bytes, settings: Settings) -> None:
        result = analyze_bytes(csv_bytes, "customers.csv", settings=settings)

        assert result.row_count == 6
        assert result.column_count == 5
        assert result.profile.column_count == 5
        assert len(result.findings) > 0
        assert 0 <= result.score.overall <= 100
        assert result.analysis_id
        assert result.duration_seconds >= 0

    def test_happy_path_xlsx(self, xlsx_bytes: bytes, settings: Settings) -> None:
        result = analyze_bytes(xlsx_bytes, "customers.xlsx", settings=settings)
        assert result.row_count == 6
        assert len(result.findings) > 0

    def test_finds_the_seeded_problems(self, csv_bytes: bytes, settings: Settings) -> None:
        result = analyze_bytes(csv_bytes, "customers.csv", settings=settings)
        issue_types = {finding.issue_type for finding in result.findings.findings}

        assert IssueType.MISSING_VALUES in issue_types
        assert IssueType.EXACT_DUPLICATE_ROWS in issue_types
        assert IssueType.LEADING_TRAILING_WHITESPACE in issue_types
        assert IssueType.INCONSISTENT_CAPITALIZATION in issue_types
        assert IssueType.CONSTANT_COLUMN in issue_types

    def test_every_finding_is_explained(self, csv_bytes: bytes, settings: Settings) -> None:
        result = analyze_bytes(csv_bytes, "customers.csv", settings=settings)
        for finding in result.findings.findings:
            assert finding.title
            assert len(finding.explanation) > 20
            assert finding.finding_id

    def test_finding_ids_are_stable_across_runs(self, csv_bytes: bytes, settings: Settings) -> None:
        first = analyze_bytes(csv_bytes, "customers.csv", settings=settings)
        second = analyze_bytes(csv_bytes, "customers.csv", settings=settings)
        assert [f.finding_id for f in first.findings.sorted()] == [
            f.finding_id for f in second.findings.sorted()
        ]
        assert first.score.overall == second.score.overall

    def test_summary_contains_no_cell_values(self, csv_bytes: bytes, settings: Settings) -> None:
        result = analyze_bytes(csv_bytes, "customers.csv", settings=settings)
        summary = str(result.summary())
        assert "Alice" not in summary
        assert "Chlo" not in summary

    def test_invalid_input_raises_typed_error(self, settings: Settings) -> None:
        with pytest.raises(UnsupportedFileTypeError):
            analyze_bytes(b"{}", "data.json", settings=settings)
