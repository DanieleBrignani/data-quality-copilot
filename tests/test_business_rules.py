from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pytest

from dqcopilot.models import IssueType, Severity
from dqcopilot.profiling import profile_dataset
from dqcopilot.rules import RuleConfigError, load_rules, load_rules_or_none, parse_rules
from dqcopilot.rules.models import BusinessRuleSet
from dqcopilot.validation import CheckContext, run_checks

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RULES_FILE = PROJECT_ROOT / "config" / "business_rules.yaml"


def evaluate(frame: pd.DataFrame, rules: BusinessRuleSet) -> list:
    """Run only the business rule check and return its findings."""
    profile = profile_dataset(frame, "t")
    context = CheckContext(frame=frame, profile=profile, rules=rules)
    findings = run_checks(context, only=["business_rule"])
    return [f for f in findings.findings if f.severity is not Severity.INFO]


def rule_set(*rules: dict) -> BusinessRuleSet:
    return parse_rules({"version": 1, "name": "t", "rules": list(rules)})


class TestRuleParsing:
    def test_loads_the_project_rule_file(self) -> None:
        rules = load_rules(RULES_FILE)
        assert rules.name == "demo-default"
        assert len(rules.rules) > 10
        assert all(rule.name for rule in rules.rules)

    def test_missing_file_raises(self) -> None:
        with pytest.raises(RuleConfigError, match="not found"):
            load_rules(PROJECT_ROOT / "config" / "does_not_exist.yaml")

    def test_missing_file_returns_none_in_lenient_mode(self) -> None:
        assert load_rules_or_none(PROJECT_ROOT / "nope.yaml") is None

    def test_invalid_yaml_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.yaml"
        path.write_text("rules: [\n  - unclosed", encoding="utf-8")
        with pytest.raises(RuleConfigError, match="YAML"):
            load_rules(path)

    def test_unknown_rule_type_is_rejected(self) -> None:
        with pytest.raises(RuleConfigError):
            rule_set({"name": "x", "type": "telepathy", "column": "a"})

    def test_duplicate_rule_names_are_rejected(self) -> None:
        with pytest.raises(RuleConfigError):
            rule_set(
                {"name": "dup", "type": "not_null", "column": "a"},
                {"name": "dup", "type": "not_null", "column": "b"},
            )

    def test_range_without_bounds_is_rejected(self) -> None:
        with pytest.raises(RuleConfigError):
            rule_set({"name": "x", "type": "range", "column": "a"})

    def test_inverted_range_is_rejected(self) -> None:
        with pytest.raises(RuleConfigError):
            rule_set({"name": "x", "type": "range", "column": "a", "minimum": 10, "maximum": 1})

    def test_invalid_regex_is_rejected(self) -> None:
        with pytest.raises(RuleConfigError):
            rule_set({"name": "x", "type": "regex", "column": "a", "pattern": "([unclosed"})

    def test_empty_file_yields_empty_rule_set(self, tmp_path: Path) -> None:
        path = tmp_path / "empty.yaml"
        path.write_text("", encoding="utf-8")
        assert load_rules(path).rules == []


class TestRuleApplicability:
    def test_rules_for_absent_columns_are_skipped(self) -> None:
        rules = rule_set(
            {"name": "present", "type": "not_null", "column": "a"},
            {"name": "absent", "type": "not_null", "column": "zzz"},
        )
        applicable, skipped = rules.applicable({"a"})
        assert [rule.name for rule in applicable] == ["present"]
        assert [rule.name for rule in skipped] == ["absent"]

    def test_disabled_rules_are_ignored(self) -> None:
        rules = rule_set({"name": "off", "type": "not_null", "column": "a", "enabled": False})
        applicable, skipped = rules.applicable({"a"})
        assert not applicable and not skipped

    def test_skipped_rules_produce_one_informational_finding(self) -> None:
        frame = pd.DataFrame({"a": ["x", "y"]})
        rules = rule_set({"name": "absent", "type": "not_null", "column": "zzz"})
        profile = profile_dataset(frame, "t")
        findings = run_checks(
            CheckContext(frame=frame, profile=profile, rules=rules), only=["business_rule"]
        )
        assert len(findings) == 1
        assert findings.findings[0].severity is Severity.INFO
        assert "absent" in findings.findings[0].explanation

    def test_no_rules_means_no_findings(self) -> None:
        frame = pd.DataFrame({"a": [None, "y"]})
        profile = profile_dataset(frame, "t")
        findings = run_checks(
            CheckContext(frame=frame, profile=profile, rules=None), only=["business_rule"]
        )
        assert len(findings) == 0


class TestNotNullRule:
    def test_flags_missing_and_blank(self) -> None:
        frame = pd.DataFrame({"a": ["x", None, "   ", "y"]})
        rules = rule_set({"name": "req", "type": "not_null", "column": "a"})
        findings = evaluate(frame, rules)
        assert findings[0].affected_rows == 2
        assert findings[0].severity is Severity.CRITICAL

    def test_clean_column_produces_nothing(self) -> None:
        frame = pd.DataFrame({"a": ["x", "y"]})
        assert evaluate(frame, rule_set({"name": "r", "type": "not_null", "column": "a"})) == []


class TestRangeRule:
    def test_flags_values_below_minimum(self) -> None:
        frame = pd.DataFrame({"revenue": ["100", "-5", "0", "-99"]})
        rules = rule_set({"name": "r", "type": "range", "column": "revenue", "minimum": 0})
        assert evaluate(frame, rules)[0].affected_rows == 2

    def test_flags_values_above_maximum(self) -> None:
        frame = pd.DataFrame({"age": ["30", "150", "121", "120"]})
        rules = rule_set(
            {"name": "r", "type": "range", "column": "age", "minimum": 0, "maximum": 120}
        )
        assert evaluate(frame, rules)[0].affected_rows == 2

    def test_bounds_are_inclusive(self) -> None:
        frame = pd.DataFrame({"age": ["0", "120"]})
        rules = rule_set(
            {"name": "r", "type": "range", "column": "age", "minimum": 0, "maximum": 120}
        )
        assert evaluate(frame, rules) == []

    def test_parses_decorated_numbers(self) -> None:
        frame = pd.DataFrame({"revenue": ["€ 1.234,56", "-1,000.00"]})
        rules = rule_set({"name": "r", "type": "range", "column": "revenue", "minimum": 0})
        assert evaluate(frame, rules)[0].affected_rows == 1

    def test_unparseable_values_are_not_violations(self) -> None:
        frame = pd.DataFrame({"revenue": ["twelve", None]})
        rules = rule_set({"name": "r", "type": "range", "column": "revenue", "minimum": 0})
        assert evaluate(frame, rules) == []


class TestAllowedValuesRule:
    def test_flags_values_outside_the_vocabulary(self) -> None:
        frame = pd.DataFrame({"country": ["FR", "DE", "France", "XX"]})
        rules = rule_set(
            {"name": "r", "type": "allowed_values", "column": "country", "values": ["FR", "DE"]}
        )
        assert evaluate(frame, rules)[0].affected_rows == 2

    def test_is_case_insensitive_by_default(self) -> None:
        frame = pd.DataFrame({"country": ["fr", " DE "]})
        rules = rule_set(
            {"name": "r", "type": "allowed_values", "column": "country", "values": ["FR", "DE"]}
        )
        assert evaluate(frame, rules) == []

    def test_case_sensitive_mode(self) -> None:
        frame = pd.DataFrame({"country": ["fr", "DE"]})
        rules = rule_set(
            {
                "name": "r",
                "type": "allowed_values",
                "column": "country",
                "values": ["FR", "DE"],
                "case_sensitive": True,
            }
        )
        assert evaluate(frame, rules)[0].affected_rows == 1

    def test_missing_values_are_not_vocabulary_violations(self) -> None:
        frame = pd.DataFrame({"country": ["FR", None, "  "]})
        rules = rule_set(
            {"name": "r", "type": "allowed_values", "column": "country", "values": ["FR"]}
        )
        assert evaluate(frame, rules) == []


class TestNoFutureDatesRule:
    def test_flags_future_dates(self) -> None:
        future = (date.today() + timedelta(days=10)).isoformat()
        frame = pd.DataFrame({"signup_date": ["2024-01-01", future]})
        rules = rule_set({"name": "r", "type": "no_future_dates", "column": "signup_date"})
        assert evaluate(frame, rules)[0].affected_rows == 1

    def test_tolerance_allows_near_future(self) -> None:
        future = (date.today() + timedelta(days=3)).isoformat()
        frame = pd.DataFrame({"signup_date": [future]})
        rules = rule_set(
            {
                "name": "r",
                "type": "no_future_dates",
                "column": "signup_date",
                "tolerance_days": 7,
            }
        )
        assert evaluate(frame, rules) == []


class TestRegexRule:
    def test_flags_values_that_do_not_match(self) -> None:
        frame = pd.DataFrame({"vat_number": ["FR12345678", "123", "de987654321"]})
        rules = rule_set(
            {
                "name": "r",
                "type": "regex",
                "column": "vat_number",
                "pattern": "^[A-Z]{2}[A-Z0-9]{8,12}$",
            }
        )
        assert evaluate(frame, rules)[0].affected_rows == 2

    def test_missing_values_are_skipped(self) -> None:
        frame = pd.DataFrame({"vat_number": [None, "  "]})
        rules = rule_set(
            {"name": "r", "type": "regex", "column": "vat_number", "pattern": "^[A-Z]{2}"}
        )
        assert evaluate(frame, rules) == []


class TestUniqueRule:
    def test_flags_repeated_values(self) -> None:
        frame = pd.DataFrame({"customer_id": ["A", "B", "A", "C", "a"]})
        rules = rule_set({"name": "r", "type": "unique", "columns": ["customer_id"]})
        # "A" repeats twice (rows 2 and 4, the latter differing only by case).
        assert evaluate(frame, rules)[0].affected_rows == 2

    def test_composite_key(self) -> None:
        frame = pd.DataFrame({"a": ["x", "x", "x"], "b": ["1", "2", "1"]})
        rules = rule_set({"name": "r", "type": "unique", "columns": ["a", "b"]})
        assert evaluate(frame, rules)[0].affected_rows == 1


class TestComparisonRule:
    def test_flags_delivery_before_order(self) -> None:
        frame = pd.DataFrame(
            {
                "order_date": ["2024-01-10", "2024-01-10"],
                "delivery_date": ["2024-01-15", "2024-01-05"],
            }
        )
        rules = rule_set(
            {
                "name": "r",
                "type": "comparison",
                "left": "delivery_date",
                "operator": ">=",
                "right": "order_date",
                "compare_as": "date",
            }
        )
        findings = evaluate(frame, rules)
        assert findings[0].affected_rows == 1
        assert findings[0].column is None  # spans two columns

    def test_numeric_comparison(self) -> None:
        frame = pd.DataFrame({"low": ["1", "9"], "high": ["5", "5"]})
        rules = rule_set(
            {
                "name": "r",
                "type": "comparison",
                "left": "low",
                "operator": "<=",
                "right": "high",
                "compare_as": "numeric",
            }
        )
        assert evaluate(frame, rules)[0].affected_rows == 1

    def test_rows_with_unparseable_values_are_skipped(self) -> None:
        frame = pd.DataFrame({"low": ["abc"], "high": ["5"]})
        rules = rule_set(
            {
                "name": "r",
                "type": "comparison",
                "left": "low",
                "operator": "<=",
                "right": "high",
                "compare_as": "numeric",
            }
        )
        assert evaluate(frame, rules) == []


class TestFindingShape:
    def test_finding_carries_the_rule_metadata(self) -> None:
        frame = pd.DataFrame({"revenue": ["-5"]})
        rules = rule_set(
            {
                "name": "revenue_not_negative",
                "type": "range",
                "column": "revenue",
                "minimum": 0,
                "description": "Revenue is recorded gross.",
                "severity": "high",
            }
        )
        finding = evaluate(frame, rules)[0]
        assert finding.issue_type is IssueType.BUSINESS_RULE_VIOLATION
        assert finding.details["rule_name"] == "revenue_not_negative"
        assert finding.details["rule_type"] == "range"
        assert finding.severity is Severity.HIGH
        assert "Revenue is recorded gross." in finding.explanation

    def test_finding_ids_differ_per_rule(self) -> None:
        frame = pd.DataFrame({"a": ["-1"], "b": ["-1"]})
        rules = rule_set(
            {"name": "rule_a", "type": "range", "column": "a", "minimum": 0},
            {"name": "rule_b", "type": "range", "column": "b", "minimum": 0},
        )
        findings = evaluate(frame, rules)
        assert len({finding.finding_id for finding in findings}) == 2


class TestRulesFileResolution:
    """Regression guard for a bug that only appeared inside the Docker image.

    ``PROJECT_ROOT`` is derived from the package location. With an editable install it
    is the repository root; in a built image the package lives under ``site-packages``,
    so the derived path pointed at a directory that does not contain the rule file and
    the business rules silently stopped running.
    """

    def test_default_resolves_to_the_project_rule_file(self) -> None:
        from dqcopilot.config import Settings as FreshSettings

        settings = FreshSettings(anthropic_api_key=None)
        assert settings.rules_file.is_file(), (
            f"the default rule file must resolve to a real file, got {settings.rules_file}"
        )
        assert load_rules(settings.rules_file).name == "demo-default"

    def test_relative_path_resolves_against_the_working_directory(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from dqcopilot.config import resolve_config_path

        (tmp_path / "config").mkdir()
        target = tmp_path / "config" / "business_rules.yaml"
        target.write_text("version: 1\nname: local\nrules: []\n", encoding="utf-8")

        monkeypatch.chdir(tmp_path)
        assert resolve_config_path(Path("config/business_rules.yaml")) == target.resolve()

    def test_absolute_path_is_taken_as_given(self, tmp_path: Path) -> None:
        from dqcopilot.config import resolve_config_path

        absolute = tmp_path / "elsewhere.yaml"
        assert resolve_config_path(absolute) == absolute

    def test_unresolvable_path_names_something_recognisable(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The error must not point at a site-packages directory nobody recognises."""
        from dqcopilot.config import resolve_config_path

        monkeypatch.chdir(tmp_path)
        result = resolve_config_path(Path("nope/missing.yaml"))
        assert result == tmp_path / "nope" / "missing.yaml"
        assert "site-packages" not in str(result)
