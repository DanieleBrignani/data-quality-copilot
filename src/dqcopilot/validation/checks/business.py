"""The configurable business rule engine, expressed as a check."""

from __future__ import annotations

import re
from datetime import date, timedelta

import pandas as pd

from dqcopilot.models.enums import FindingSource, IssueType, Severity
from dqcopilot.models.findings import Finding, make_finding_id
from dqcopilot.profiling.type_inference import (
    coerce_numeric,
    missing_mask,
    parse_dates_with_format,
    to_clean_strings,
)
from dqcopilot.rules.models import (
    AllowedValuesRule,
    BusinessRule,
    ComparisonRule,
    NoFutureDatesRule,
    NotNullRule,
    RangeRule,
    RegexRule,
    UniqueRule,
)
from dqcopilot.validation.base import Check, CheckContext, pct, sample_indices
from dqcopilot.validation.registry import register_check

_OPERATORS = {
    "<": lambda left, right: left < right,
    "<=": lambda left, right: left <= right,
    ">": lambda left, right: left > right,
    ">=": lambda left, right: left >= right,
    "==": lambda left, right: left == right,
    "!=": lambda left, right: left != right,
}


@register_check
class BusinessRuleCheck(Check):
    """Evaluate every applicable business rule from the loaded rule set.

    Rules that reference a column the dataset does not have are skipped, not failed:
    one rule file is meant to serve several datasets. Skipped rules are reported once,
    as a single informational finding, so nothing disappears silently.
    """

    check_id = "business_rule"
    title = "Business rules"
    description = "Evaluates the configurable rules declared in the YAML rule file."

    def run(self, context: CheckContext) -> list[Finding]:  # noqa: D102 - inherited
        if context.rules is None:
            return []

        columns = {column.name for column in context.columns()}
        applicable, skipped = context.rules.applicable(columns)

        findings: list[Finding] = []
        for rule in applicable:
            finding = self._evaluate(rule, context)
            if finding is not None:
                findings.append(finding)

        if skipped:
            findings.append(self._skipped_finding(skipped, context))
        return findings

    # ------------------------------------------------------------------ dispatch

    def _evaluate(self, rule: BusinessRule, context: CheckContext) -> Finding | None:
        if isinstance(rule, NotNullRule):
            mask, detail = self._not_null(rule, context)
        elif isinstance(rule, RangeRule):
            mask, detail = self._range(rule, context)
        elif isinstance(rule, AllowedValuesRule):
            mask, detail = self._allowed_values(rule, context)
        elif isinstance(rule, NoFutureDatesRule):
            mask, detail = self._no_future_dates(rule, context)
        elif isinstance(rule, RegexRule):
            mask, detail = self._regex(rule, context)
        elif isinstance(rule, UniqueRule):
            mask, detail = self._unique(rule, context)
        elif isinstance(rule, ComparisonRule):
            mask, detail = self._comparison(rule, context)
        else:  # pragma: no cover - the union is exhaustive
            return None

        affected = int(mask.sum())
        if affected == 0:
            return None

        column = rule.columns_used()[0] if len(rule.columns_used()) == 1 else None
        ratio = affected / context.row_count if context.row_count else 0.0

        return Finding(
            finding_id=make_finding_id(
                FindingSource.DETERMINISTIC,
                self.check_id,
                IssueType.BUSINESS_RULE_VIOLATION,
                column,
                fingerprint=rule.name,
            ),
            check_id=self.check_id,
            issue_type=IssueType.BUSINESS_RULE_VIOLATION,
            severity=rule.severity,
            column=column,
            title=f"Business rule failed: {rule.name}",
            explanation=(
                f"{affected:,} row(s) ({pct(ratio)}) break the rule '{rule.name}'. {detail}"
                + (f" {rule.description}" if rule.description else "")
            ),
            affected_rows=affected,
            row_count=context.row_count,
            row_indices=sample_indices(mask),
            details={
                "rule_name": rule.name,
                "rule_type": rule.type,
                "columns": rule.columns_used(),
            },
        )

    def _skipped_finding(self, skipped: list[BusinessRule], context: CheckContext) -> Finding:
        names = [rule.name for rule in skipped]
        return Finding(
            finding_id=make_finding_id(
                FindingSource.DETERMINISTIC,
                self.check_id,
                IssueType.BUSINESS_RULE_VIOLATION,
                None,
                fingerprint="__skipped__",
            ),
            check_id=self.check_id,
            issue_type=IssueType.BUSINESS_RULE_VIOLATION,
            severity=Severity.INFO,
            column=None,
            title=f"{len(skipped)} business rule(s) were not evaluated",
            explanation=(
                "These rules reference columns that this dataset does not contain, so they "
                f"were skipped rather than failed: {', '.join(names[:8])}"
                + ("…" if len(names) > 8 else "")
                + ". This is expected when one rule file covers several datasets."
            ),
            affected_rows=0,
            row_count=context.row_count,
            details={
                "skipped_rules": [
                    {"name": rule.name, "columns": rule.columns_used()} for rule in skipped[:20]
                ]
            },
        )

    # ------------------------------------------------------------------ evaluators

    @staticmethod
    def _not_null(rule: NotNullRule, context: CheckContext) -> tuple[pd.Series, str]:
        mask = missing_mask(context.series(rule.column))
        return mask, f"'{rule.column}' is mandatory but these rows leave it empty."

    @staticmethod
    def _range(rule: RangeRule, context: CheckContext) -> tuple[pd.Series, str]:
        numeric = coerce_numeric(context.series(rule.column))
        mask = pd.Series(False, index=numeric.index)
        if rule.minimum is not None:
            mask |= numeric < rule.minimum
        if rule.maximum is not None:
            mask |= numeric > rule.maximum
        mask = mask.fillna(False).astype(bool)
        return mask, f"'{rule.column}' must be {rule.describe_bounds()}."

    @staticmethod
    def _allowed_values(rule: AllowedValuesRule, context: CheckContext) -> tuple[pd.Series, str]:
        series = context.series(rule.column)
        text = to_clean_strings(series)
        present = ~missing_mask(series)
        normalised = text.map(lambda value: rule.normalise(str(value)), na_action="ignore")
        known = normalised.isin(rule.allowed_set()).astype("boolean").fillna(False).astype(bool)
        mask = present & ~known

        vocabulary = ", ".join(repr(value) for value in rule.values[:8])
        suffix = "…" if len(rule.values) > 8 else ""
        return mask.astype(bool), (
            f"'{rule.column}' only accepts: {vocabulary}{suffix}"
            + ("" if rule.case_sensitive else " (case-insensitive)")
            + "."
        )

    @staticmethod
    def _no_future_dates(rule: NoFutureDatesRule, context: CheckContext) -> tuple[pd.Series, str]:
        limit = pd.Timestamp(date.today() + timedelta(days=rule.tolerance_days))
        parsed = parse_dates_with_format(context.series(rule.column), None)
        mask = (parsed > limit).fillna(False).astype(bool)
        tolerance = (
            f" (a tolerance of {rule.tolerance_days} day(s) is allowed)"
            if rule.tolerance_days
            else ""
        )
        return mask, f"'{rule.column}' must not be later than {limit.date()}{tolerance}."

    @staticmethod
    def _regex(rule: RegexRule, context: CheckContext) -> tuple[pd.Series, str]:
        pattern = re.compile(rule.pattern)
        series = context.series(rule.column)
        text = to_clean_strings(series)
        present = ~missing_mask(series)
        matches = (
            text.map(lambda value: bool(pattern.match(str(value))), na_action="ignore")
            .astype("boolean")
            .fillna(False)
            .astype(bool)
        )
        mask = present & ~matches
        return mask.astype(bool), f"'{rule.column}' must match the pattern `{rule.pattern}`."

    @staticmethod
    def _unique(rule: UniqueRule, context: CheckContext) -> tuple[pd.Series, str]:
        subset = context.frame[rule.columns]
        normalised = subset.astype("string").apply(lambda col: col.str.strip().str.casefold())
        mask = normalised.duplicated(keep="first")
        joined = " + ".join(f"'{column}'" for column in rule.columns)
        return mask.astype(bool), f"{joined} must be unique but these rows repeat a value."

    @staticmethod
    def _comparison(rule: ComparisonRule, context: CheckContext) -> tuple[pd.Series, str]:
        if rule.compare_as == "date":
            left = parse_dates_with_format(context.series(rule.left), None)
            right = parse_dates_with_format(context.series(rule.right), None)
        else:
            left = coerce_numeric(context.series(rule.left))
            right = coerce_numeric(context.series(rule.right))

        comparable = left.notna() & right.notna()
        satisfied = _OPERATORS[rule.operator](left, right)
        mask = comparable & ~satisfied.fillna(False).astype(bool)
        return mask.astype(bool), (
            f"'{rule.left}' must be {rule.operator} '{rule.right}' (compared as {rule.compare_as})."
        )
