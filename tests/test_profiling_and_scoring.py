from __future__ import annotations

import pandas as pd

from dqcopilot.models import FindingSource, IssueType, SemanticType, Severity
from dqcopilot.models.findings import Finding, FindingSet, make_finding_id
from dqcopilot.profiling import profile_dataset
from dqcopilot.scoring import compute_quality_score


class TestProfiler:
    def test_profiles_every_column(self, messy_frame: pd.DataFrame) -> None:
        profile = profile_dataset(messy_frame, "messy")
        assert profile.column_count == 5
        assert profile.row_count == 6
        assert [c.name for c in profile.columns] == list(messy_frame.columns)
        assert all(c.position == i for i, c in enumerate(profile.columns))

    def test_counts_missing_including_blanks(self, messy_frame: pd.DataFrame) -> None:
        profile = profile_dataset(messy_frame, "messy")
        notes = profile.column("notes")
        assert notes is not None
        assert notes.missing_count == 5  # "", "  ", None, None, "  "

    def test_counts_duplicate_rows(self) -> None:
        frame = pd.DataFrame({"a": ["x", "y", "x"]})
        assert profile_dataset(frame, "d").duplicate_row_count == 1

    def test_numeric_stats(self) -> None:
        frame = pd.DataFrame({"revenue": ["100", "-50", "0", "1,000"]})
        column = profile_dataset(frame, "n").column("revenue")
        assert column is not None
        assert column.semantic_type is SemanticType.INTEGER
        assert column.numeric is not None
        assert column.numeric.minimum == -50.0
        assert column.numeric.maximum == 1000.0
        assert column.numeric.negative_count == 1
        assert column.numeric.zero_count == 1

    def test_text_stats_flag_padding(self) -> None:
        frame = pd.DataFrame({"city": ["Paris", " Lyon", "Nice "]})
        column = profile_dataset(frame, "t").column("city")
        assert column is not None
        assert column.text is not None
        assert column.text.whitespace_padded_count == 2

    def test_unique_count_ignores_case_padding_difference(self) -> None:
        frame = pd.DataFrame({"country": ["FR", " FR", "DE"]})
        column = profile_dataset(frame, "c").column("country")
        assert column is not None
        assert column.unique_count == 2  # "FR" and "DE" after trimming

    def test_dataset_level_ratios(self) -> None:
        frame = pd.DataFrame({"a": ["x", None], "b": ["y", "z"]})
        profile = profile_dataset(frame, "r")
        assert profile.total_cells == 4
        assert profile.total_missing == 1
        assert profile.missing_ratio == 0.25


def _finding(column: str | None, severity: Severity, affected: int, rows: int = 100) -> Finding:
    return Finding(
        finding_id=make_finding_id(
            FindingSource.DETERMINISTIC, "t", IssueType.MISSING_VALUES, column, str(severity)
        ),
        check_id="t",
        issue_type=IssueType.MISSING_VALUES,
        severity=severity,
        column=column,
        title="t",
        explanation="t",
        affected_rows=affected,
        row_count=rows,
    )


class TestQualityScore:
    def test_perfect_dataset_scores_100(self) -> None:
        frame = pd.DataFrame({"a": ["x", "y"], "b": ["1", "2"]})
        profile = profile_dataset(frame, "p")
        score = compute_quality_score(profile, FindingSet(findings=[]))
        assert score.overall == 100.0
        assert score.grade == "A"

    def test_penalty_matches_documented_formula(self) -> None:
        frame = pd.DataFrame({"a": ["x"] * 100, "b": ["y"] * 100})
        profile = profile_dataset(frame, "p")
        # critical (weight 1.0) affecting 50% of rows -> penalty 50 on column "a"
        findings = FindingSet(findings=[_finding("a", Severity.CRITICAL, 50)])
        score = compute_quality_score(profile, findings)
        assert score.column_score("a") == 50.0
        assert score.column_score("b") == 100.0
        assert score.overall == 75.0  # mean(50, 100)

    def test_dataset_level_penalty_applies_to_overall(self) -> None:
        frame = pd.DataFrame({"a": ["x"] * 100})
        profile = profile_dataset(frame, "p")
        findings = FindingSet(findings=[_finding(None, Severity.HIGH, 50)])
        score = compute_quality_score(profile, findings)
        assert score.dataset_penalty == 30.0  # 0.6 * 0.5 * 100
        assert score.overall == 70.0

    def test_score_never_goes_below_zero(self) -> None:
        frame = pd.DataFrame({"a": ["x"] * 10})
        profile = profile_dataset(frame, "p")
        findings = FindingSet(
            findings=[
                _finding("a", Severity.CRITICAL, 10, rows=10),
                _finding(None, Severity.CRITICAL, 10, rows=10),
            ]
        )
        assert compute_quality_score(profile, findings).overall == 0.0

    def test_ai_findings_do_not_change_the_score(self) -> None:
        frame = pd.DataFrame({"a": ["x"] * 100})
        profile = profile_dataset(frame, "p")
        ai_finding = _finding("a", Severity.CRITICAL, 100)
        ai_finding.source = FindingSource.AI
        score = compute_quality_score(profile, FindingSet(findings=[ai_finding]))
        assert score.overall == 100.0

    def test_info_findings_are_free(self) -> None:
        frame = pd.DataFrame({"a": ["x"] * 100})
        profile = profile_dataset(frame, "p")
        findings = FindingSet(findings=[_finding("a", Severity.INFO, 100)])
        assert compute_quality_score(profile, findings).overall == 100.0
