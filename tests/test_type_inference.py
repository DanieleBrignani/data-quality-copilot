from __future__ import annotations

import pandas as pd
import pytest

from dqcopilot.models import SemanticType
from dqcopilot.profiling.type_inference import (
    analyze_dates,
    coerce_numeric,
    infer_semantic_type,
    match_date_formats,
    missing_mask,
    parse_boolean_token,
    parse_numeric_token,
)


class TestParseNumericToken:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("1234", 1234.0),
            ("  42 ", 42.0),
            ("-42.5", -42.5),
            ("+7", 7.0),
            ("1,234", 1234.0),
            ("1,234.56", 1234.56),
            ("1.234,56", 1234.56),
            ("1.234.567", 1234567.0),
            ("1,5", 1.5),
            ("$1,234.56", 1234.56),
            ("€ 900", 900.0),
            ("(500)", -500.0),
            ("15%", 15.0),
            ("1e3", 1000.0),
        ],
    )
    def test_parses_human_written_numbers(self, raw: str, expected: float) -> None:
        assert parse_numeric_token(raw) == pytest.approx(expected)

    @pytest.mark.parametrize("raw", ["", "   ", "twelve", "N/A", "12abc", "-", "1.2.3,4,5"])
    def test_rejects_non_numeric(self, raw: str) -> None:
        assert parse_numeric_token(raw) is None


class TestCoerceNumeric:
    def test_leaves_unparseable_as_nan(self) -> None:
        series = pd.Series(["1,234.56", "€900", "twelve", None])
        result = coerce_numeric(series)
        assert result.tolist()[:2] == [1234.56, 900.0]
        assert pd.isna(result.iloc[2])
        assert pd.isna(result.iloc[3])

    def test_passes_native_numeric_through(self) -> None:
        series = pd.Series([1, 2, 3])
        assert coerce_numeric(series).tolist() == [1.0, 2.0, 3.0]


class TestMatchDateFormats:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("2024-01-15", {"%Y-%m-%d"}),
            ("15/01/2024", {"%d/%m/%Y"}),
            ("01/15/2024", {"%m/%d/%Y"}),
            ("20240115", {"%Y%m%d"}),
        ],
    )
    def test_matches_expected_formats(self, raw: str, expected: set[str]) -> None:
        assert set(match_date_formats(raw)) == expected

    def test_ambiguous_value_matches_several_formats(self) -> None:
        matches = match_date_formats("03/04/2024")
        assert "%d/%m/%Y" in matches
        assert "%m/%d/%Y" in matches

    @pytest.mark.parametrize("raw", ["2024", "31/02/2024", "2024-13-01", "not a date", ""])
    def test_rejects_invalid_dates(self, raw: str) -> None:
        assert match_date_formats(raw) == frozenset()


class TestAnalyzeDates:
    def test_consistent_column(self) -> None:
        series = pd.Series(["2024-01-15", "2024-02-20", "2023-12-31"])
        analysis = analyze_dates(series)
        assert analysis.is_format_consistent
        assert analysis.consistent_formats == frozenset({"%Y-%m-%d"})
        assert analysis.parse_ratio == 1.0

    def test_mixed_formats_are_not_consistent(self) -> None:
        series = pd.Series(["2024-01-15", "15/01/2024", "Jan 15, 2024"])
        analysis = analyze_dates(series)
        assert not analysis.is_format_consistent

    def test_reports_unparseable_rows(self) -> None:
        series = pd.Series(["2024-01-15", "31/02/2024", "2024-02-20"])
        analysis = analyze_dates(series)
        assert analysis.unparseable_indices == [1]
        assert analysis.parsed_count == 2


class TestInferSemanticType:
    @pytest.mark.parametrize(
        ("values", "expected"),
        [
            (["a@b.com", "c@d.org", "e@f.net"], SemanticType.EMAIL),
            (["2024-01-15", "2024-02-20", "2023-12-31"], SemanticType.DATE),
            (["1", "2", "3", "4"], SemanticType.INTEGER),
            (["1.5", "2.25", "3.75"], SemanticType.FLOAT),
            (["yes", "no", "yes", "no"], SemanticType.BOOLEAN),
            (["FR", "DE", "FR", "IT", "FR"], SemanticType.CATEGORICAL),
            ([None, None, None], SemanticType.EMPTY),
        ],
    )
    def test_infers_expected_type(self, values: list[str | None], expected: SemanticType) -> None:
        kind, _ = infer_semantic_type(pd.Series(values, dtype=object), "col")
        assert kind is expected

    def test_uses_native_pandas_types(self) -> None:
        assert infer_semantic_type(pd.Series([1, 2, 3]), "n")[0] is SemanticType.INTEGER
        assert infer_semantic_type(pd.Series([True, False]), "b")[0] is SemanticType.BOOLEAN
        dates = pd.Series(pd.to_datetime(["2024-01-01", "2024-01-02"]))
        assert infer_semantic_type(dates, "d")[0] is SemanticType.DATETIME

    def test_numeric_stored_as_text_is_still_numeric(self) -> None:
        series = pd.Series(["1,234.56", "€900", "-50", "0", "77"])
        kind, confidence = infer_semantic_type(series, "revenue")
        assert kind is SemanticType.FLOAT
        assert confidence == 1.0

    def test_identifier_detection(self) -> None:
        series = pd.Series([f"C{i:04d}" for i in range(20)])
        assert infer_semantic_type(series, "customer_id")[0] is SemanticType.IDENTIFIER


class TestMissingMask:
    def test_treats_blank_strings_as_missing(self) -> None:
        series = pd.Series(["a", "", "   ", None, "b"])
        assert missing_mask(series).tolist() == [False, True, True, True, False]

    def test_numeric_column(self) -> None:
        series = pd.Series([1.0, float("nan"), 3.0])
        assert missing_mask(series).tolist() == [False, True, False]


class TestParseBooleanToken:
    @pytest.mark.parametrize("raw", ["true", "TRUE", "yes", "Y", "1"])
    def test_true_tokens(self, raw: str) -> None:
        assert parse_boolean_token(raw) is True

    @pytest.mark.parametrize("raw", ["false", "No", "n", "0"])
    def test_false_tokens(self, raw: str) -> None:
        assert parse_boolean_token(raw) is False

    @pytest.mark.parametrize("raw", ["maybe", "", "2"])
    def test_unknown_tokens(self, raw: str) -> None:
        assert parse_boolean_token(raw) is None
