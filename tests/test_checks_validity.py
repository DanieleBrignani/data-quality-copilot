from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import pytest

from dqcopilot.models import IssueType, Severity
from dqcopilot.profiling import profile_dataset
from dqcopilot.validation import CheckContext, run_checks
from dqcopilot.validation.base import column_tokens, name_matches


def analyse(frame: pd.DataFrame, only: list[str] | None = None) -> dict[IssueType, list]:
    profile = profile_dataset(frame, "test")
    findings = run_checks(CheckContext(frame=frame, profile=profile), only=only)
    grouped: dict[IssueType, list] = {}
    for finding in findings.findings:
        grouped.setdefault(finding.issue_type, []).append(finding)
    return grouped


class TestColumnNameMatching:
    def test_splits_on_punctuation(self) -> None:
        assert column_tokens("total_amount_2023") == {"total", "amount", "2023"}
        assert column_tokens("Delivery Date") == {"delivery", "date"}

    @pytest.mark.parametrize(
        ("column", "hints", "expected"),
        [
            ("total_amount", ("amount",), True),
            ("discount_percent", ("count",), False),  # substring trap
            ("average_price", ("age",), False),  # substring trap
            ("age", ("age",), True),
            ("customer_age", ("age",), True),
            ("account_id", ("count",), False),
        ],
    )
    def test_matches_whole_words_only(
        self, column: str, hints: tuple[str, ...], expected: bool
    ) -> None:
        assert name_matches(column, hints) is expected


class TestInvalidDate:
    def test_detects_impossible_calendar_dates(self) -> None:
        frame = pd.DataFrame(
            {"signup_date": ["2024-01-15", "2023-02-30", "2024-02-20", "2024-13-01"]}
        )
        found = analyse(frame)[IssueType.INVALID_DATE]
        assert found[0].affected_rows == 2
        assert found[0].row_indices == [1, 3]

    def test_clean_date_column_is_silent(self) -> None:
        frame = pd.DataFrame({"signup_date": ["2024-01-15", "2024-02-20", "2024-03-01"]})
        assert IssueType.INVALID_DATE not in analyse(frame)

    def test_missing_values_are_not_invalid_dates(self) -> None:
        frame = pd.DataFrame({"signup_date": ["2024-01-15", None, "  ", "2024-02-20"]})
        assert IssueType.INVALID_DATE not in analyse(frame)


class TestInconsistentDateFormat:
    def test_detects_mixed_formats(self) -> None:
        frame = pd.DataFrame(
            {"signup_date": ["2024-01-15", "15/03/2024", "2024-02-20", "20/04/2024"]}
        )
        found = analyse(frame)[IssueType.INCONSISTENT_DATE_FORMAT]
        assert found[0].severity is Severity.HIGH
        assert "format_counts" in found[0].details

    def test_single_format_is_silent(self) -> None:
        frame = pd.DataFrame({"signup_date": ["15/03/2024", "20/04/2024", "01/05/2024"]})
        assert IssueType.INCONSISTENT_DATE_FORMAT not in analyse(frame)

    def test_ambiguous_but_consistent_column_is_silent(self) -> None:
        """Values that all match both DD/MM and MM/DD are consistent, not mixed."""
        frame = pd.DataFrame({"signup_date": ["01/02/2024", "03/04/2024", "05/06/2024"]})
        assert IssueType.INCONSISTENT_DATE_FORMAT not in analyse(frame)


class TestFutureDate:
    def test_detects_future_dates_in_past_event_columns(self) -> None:
        future = (date.today() + timedelta(days=365)).isoformat()
        frame = pd.DataFrame({"signup_date": ["2024-01-15", future, "2024-02-20"]})
        found = analyse(frame)[IssueType.FUTURE_DATE]
        assert found[0].affected_rows == 1

    def test_ignores_columns_that_may_hold_future_dates(self) -> None:
        future = (date.today() + timedelta(days=365)).isoformat()
        frame = pd.DataFrame({"contract_end_date": ["2024-01-15", future, "2024-02-20"]})
        assert IssueType.FUTURE_DATE not in analyse(frame)

    def test_recognises_onboarding_columns(self) -> None:
        future = (date.today() + timedelta(days=90)).isoformat()
        frame = pd.DataFrame({"onboarded_on": ["2021-01-15", future, "2022-02-20"]})
        assert IssueType.FUTURE_DATE in analyse(frame)


class TestInvalidEmail:
    @pytest.mark.parametrize(
        "bad",
        ["alice@", "bob.example.com", "chloe@@example.com", "dave@example", "e f@example.com"],
    )
    def test_detects_malformed_addresses(self, bad: str) -> None:
        frame = pd.DataFrame({"email": ["a@example.com", "b@example.com", "c@example.com", bad]})
        found = analyse(frame)[IssueType.INVALID_EMAIL]
        assert found[0].affected_rows == 1
        assert found[0].row_indices == [3]

    def test_valid_addresses_are_silent(self) -> None:
        frame = pd.DataFrame({"email": ["a@example.com", "b.c+tag@sub.example.co.uk"]})
        assert IssueType.INVALID_EMAIL not in analyse(frame)

    def test_missing_addresses_are_not_invalid(self) -> None:
        frame = pd.DataFrame({"email": ["a@example.com", None, "  ", "b@example.com"]})
        assert IssueType.INVALID_EMAIL not in analyse(frame)


class TestNumericStoredAsText:
    def test_detects_decorated_numbers(self) -> None:
        frame = pd.DataFrame({"revenue": ["1,234.56", "€900", "(500)", "77"]})
        found = analyse(frame)[IssueType.NUMERIC_STORED_AS_TEXT]
        assert found[0].details["decorated"] == 3

    def test_plain_numeric_strings_are_not_flagged(self) -> None:
        """Every CSV column is text; only formatting that blocks a cast is a problem."""
        frame = pd.DataFrame({"age": ["30", "41", "52", "-7", "3.5"]})
        assert IssueType.NUMERIC_STORED_AS_TEXT not in analyse(frame)

    def test_native_numeric_columns_are_not_flagged(self) -> None:
        frame = pd.DataFrame({"revenue": [1234.56, 900.0, 77.0]})
        assert IssueType.NUMERIC_STORED_AS_TEXT not in analyse(frame)

    def test_unparseable_values_are_reported(self) -> None:
        frame = pd.DataFrame({"revenue": ["100", "200", "300", "twelve"]})
        found = analyse(frame)[IssueType.NUMERIC_STORED_AS_TEXT]
        assert found[0].details["unparseable"] == 1


class TestImpossibleNumeric:
    def test_detects_negative_amounts(self) -> None:
        frame = pd.DataFrame({"revenue": ["100", "-50", "200", "-1"]})
        found = analyse(frame)[IssueType.IMPOSSIBLE_NUMERIC]
        assert found[0].affected_rows == 2

    def test_detects_impossible_ages(self) -> None:
        frame = pd.DataFrame({"age": ["30", "-5", "199", "45"]})
        found = analyse(frame)[IssueType.IMPOSSIBLE_NUMERIC]
        assert found[0].affected_rows == 2

    def test_does_not_fire_on_unrelated_columns(self) -> None:
        frame = pd.DataFrame({"temperature_celsius": ["-10", "-5", "20"]})
        assert IssueType.IMPOSSIBLE_NUMERIC not in analyse(frame)

    def test_does_not_fire_on_discount_via_count_substring(self) -> None:
        frame = pd.DataFrame({"discount_percent": ["-5", "10", "20"]})
        assert IssueType.IMPOSSIBLE_NUMERIC not in analyse(frame)


class TestSuspiciousNumeric:
    def test_detects_extreme_outliers(self) -> None:
        values = [str(100 + i) for i in range(50)] + ["99999999"]
        frame = pd.DataFrame({"measurement": values})
        found = analyse(frame)[IssueType.SUSPICIOUS_NUMERIC]
        assert found[0].affected_rows == 1
        assert found[0].severity is Severity.LOW

    def test_needs_enough_rows(self) -> None:
        frame = pd.DataFrame({"measurement": ["1", "2", "999999"]})
        assert IssueType.SUSPICIOUS_NUMERIC not in analyse(frame)

    def test_uniform_column_has_no_outliers(self) -> None:
        frame = pd.DataFrame({"measurement": [str(100 + i) for i in range(50)]})
        assert IssueType.SUSPICIOUS_NUMERIC not in analyse(frame)


class TestInconsistentCategory:
    def test_detects_punctuation_variants(self) -> None:
        frame = pd.DataFrame(
            {"segment": ["Mid-Market", "Mid Market", "midmarket", "SMB", "SMB", "SMB"]}
        )
        found = analyse(frame)[IssueType.INCONSISTENT_CATEGORY]
        assert found[0].affected_rows == 3

    def test_case_only_differences_are_left_to_the_capitalisation_check(self) -> None:
        frame = pd.DataFrame({"country": ["FR", "fr", "DE"]})
        grouped = analyse(frame)
        assert IssueType.INCONSISTENT_CATEGORY not in grouped
        assert IssueType.INCONSISTENT_CAPITALIZATION in grouped

    def test_genuinely_different_spellings_are_not_guessed(self) -> None:
        """ "UK" and "United Kingdom" need a human or the AI panel, not a silent merge."""
        frame = pd.DataFrame({"country": ["UK", "United Kingdom", "FR", "FR"]})
        assert IssueType.INCONSISTENT_CATEGORY not in analyse(frame)


class TestMixedDatatypes:
    def test_detects_a_substantial_mix(self) -> None:
        frame = pd.DataFrame({"value": ["1", "2", "3", "4", "5", "abc", "def", "ghi"]})
        assert IssueType.MIXED_DATATYPES in analyse(frame)

    def test_small_minority_is_left_to_specific_checks(self) -> None:
        frame = pd.DataFrame({"value": [str(i) for i in range(50)] + ["abc"]})
        assert IssueType.MIXED_DATATYPES not in analyse(frame)


class TestSchemaCheck:
    def test_reports_header_repairs(self) -> None:
        frame = pd.DataFrame({"a": ["x", "y"]})
        profile = profile_dataset(frame, "t")
        context = CheckContext(
            frame=frame, profile=profile, notes=["Column header 2 was empty and was auto-named."]
        )
        findings = run_checks(context, only=["schema_mismatch"])
        assert len(findings) == 1
        assert findings.findings[0].issue_type is IssueType.SCHEMA_MISMATCH

    def test_clean_header_produces_nothing(self) -> None:
        frame = pd.DataFrame({"a": ["x", "y"]})
        profile = profile_dataset(frame, "t")
        findings = run_checks(CheckContext(frame=frame, profile=profile), only=["schema_mismatch"])
        assert len(findings) == 0


class TestRegistryRobustness:
    def test_a_failing_check_does_not_abort_the_run(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from dqcopilot.validation.checks.completeness import MissingValuesCheck

        def explode(self, context):  # noqa: ANN001, ARG001
            raise RuntimeError("boom")

        monkeypatch.setattr(MissingValuesCheck, "run", explode)
        frame = pd.DataFrame({"a": ["x", None], "b": ["  y", "z"]})
        grouped = analyse(frame)

        assert IssueType.MISSING_VALUES not in grouped
        assert IssueType.LEADING_TRAILING_WHITESPACE in grouped
