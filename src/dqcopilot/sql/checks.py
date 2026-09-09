"""Checks computed in the database instead of in Python.

Only the checks that SQL expresses *well* live here. Porting all nineteen would mean two
implementations of the same rules, free to drift apart, and the ones left out are left
out for a reason rather than for lack of time:

* **Encoding corruption** proves itself by re-encoding a string and decoding it as UTF-8.
  That is a byte-level operation, not a set operation.
* **Placeholder detection** compares a value's frequency against the column's
  distinctness, then applies a word list - a judgement, not a filter.
* **Probable duplicates** build a blocking key by stripping company legal forms.
* **Mixed types and date-format consistency** classify values against seventeen date
  formats.

What remains is what a warehouse is good at: counting, grouping and comparing. Those
answers do not need the data in Python, and this module never brings it there - every
function below returns counts and a handful of example values.

Findings carry no ``row_indices``. At this scale you do not point at rows; you report
that 3.2% of a column is negative and show twelve examples. That is the same shift the
architecture makes: review aggregates, not cells.
"""

from __future__ import annotations

from dqcopilot.logging_conf import get_logger
from dqcopilot.models.enums import FindingSource, IssueType, Severity
from dqcopilot.models.findings import Finding, FindingSet, make_finding_id
from dqcopilot.rules.models import (
    AllowedValuesRule,
    BusinessRuleSet,
    ComparisonRule,
    NotNullRule,
    RangeRule,
    RegexRule,
    UniqueRule,
)
from dqcopilot.sql.engine import (
    SqlEngineError,
    SqlSource,
    missing_expression,
    normalised_expression,
    numeric_expression,
    quote_identifier,
)
from dqcopilot.validation.base import pct, ratio_severity

logger = get_logger(__name__)

#: How many offending values to quote back in a finding.
EXAMPLE_LIMIT = 5

_OPERATOR_SQL = {"<": "<", "<=": "<=", ">": ">", ">=": ">=", "==": "=", "!=": "<>"}


def run_sql_checks(source: SqlSource, rules: BusinessRuleSet | None = None) -> FindingSet:
    """Run every SQL-expressible check against ``source``.

    A failing check is logged and skipped rather than aborting the run, matching the
    Python engine: one broken rule must not cost the user the other twelve.
    """
    findings: list[Finding] = []
    for produce in (_missing_values, _constant_columns, _exact_duplicates):
        try:
            findings.extend(produce(source))
        except SqlEngineError:
            logger.exception("SQL check failed", extra={"check": produce.__name__})

    if rules is not None:
        findings.extend(_business_rules(source, rules))
    return FindingSet(findings=findings)


# ------------------------------------------------------------------ structural checks


def _missing_values(source: SqlSource) -> list[Finding]:
    """Count empty cells per column in a single pass over the file."""
    if source.row_count == 0 or not source.columns:
        return []

    counts = ", ".join(
        f"count(*) FILTER (WHERE {missing_expression(column)})" for column in source.columns
    )
    row = source.query(f"SELECT {counts} FROM {source.table}")[0]

    findings = []
    for column, missing in zip(source.columns, row, strict=True):
        affected = int(missing or 0)
        if affected == 0:
            continue
        ratio = affected / source.row_count
        findings.append(
            _finding(
                source,
                check_id="missing_values",
                issue=IssueType.MISSING_VALUES,
                severity=ratio_severity(ratio),
                column=column,
                title=f"Missing values in '{column}'",
                explanation=(
                    f"{affected:,} of {source.row_count:,} rows ({pct(ratio)}) have no value "
                    f"in '{column}'."
                ),
                affected=affected,
            )
        )
    return findings


def _constant_columns(source: SqlSource) -> list[Finding]:
    """Report columns whose populated rows all say the same thing."""
    if source.row_count < 2 or not source.columns:
        return []

    findings = []
    for column in source.columns:
        quoted = quote_identifier(column)
        present = f"NOT {missing_expression(column)}"
        row = source.query(
            f"SELECT count(DISTINCT {quoted}), count(*) FILTER (WHERE {present}), "
            f"min({quoted}) FROM {source.table} WHERE {present}"
        )[0]
        distinct, populated, value = int(row[0] or 0), int(row[1] or 0), row[2]
        if distinct != 1 or populated == 0:
            continue
        findings.append(
            _finding(
                source,
                check_id="constant_column",
                issue=IssueType.CONSTANT_COLUMN,
                severity=Severity.INFO,
                column=column,
                title=f"'{column}' is constant",
                explanation=(
                    f"Every populated row of '{column}' holds the same value ('{value}'). "
                    "The column adds no discriminating information; this is worth "
                    "confirming rather than fixing automatically."
                ),
                affected=populated,
            )
        )
    return findings


def _exact_duplicates(source: SqlSource) -> list[Finding]:
    """Report rows repeated after trimming and case-folding, as the Python engine does."""
    if source.row_count < 2 or not source.columns:
        return []

    key = ", ".join(normalised_expression(column) for column in source.columns)
    row = source.query(
        f"SELECT coalesce(sum(n - 1), 0), count(*) FROM ("  # noqa: S608 - identifiers are quoted
        f"SELECT count(*) AS n FROM {source.table} GROUP BY {key} HAVING count(*) > 1)"
    )[0]
    affected, groups = int(row[0] or 0), int(row[1] or 0)
    if affected == 0:
        return []

    ratio = affected / source.row_count
    return [
        _finding(
            source,
            check_id="exact_duplicate_rows",
            issue=IssueType.EXACT_DUPLICATE_ROWS,
            severity=Severity.HIGH if ratio > 0.05 else Severity.MEDIUM,
            column=None,
            title="Exact duplicate rows",
            explanation=(
                f"{affected:,} of {source.row_count:,} rows ({pct(ratio)}) repeat a row that "
                f"already appears earlier in the file, across {groups:,} distinct group(s). "
                "Values are compared after trimming surrounding whitespace and ignoring "
                "letter case."
            ),
            affected=affected,
            details={"duplicate_groups": groups, "comparison": "trimmed, case-folded"},
        )
    ]


# ------------------------------------------------------------------ business rules


def _business_rules(source: SqlSource, rules: BusinessRuleSet) -> list[Finding]:
    """Evaluate every rule the SQL engine can express, and say which it skipped."""
    findings: list[Finding] = []
    skipped: list[str] = []

    for rule in rules.rules:
        if not rule.enabled:
            continue
        if any(column not in source.columns for column in rule.columns_used()):
            skipped.append(rule.name)
            continue

        built = _rule_predicate(rule, source.table)
        if built is None:
            skipped.append(rule.name)
            continue

        count_sql, parameters, sentence = built
        try:
            affected = int(source.scalar(count_sql, parameters) or 0)
        except SqlEngineError:
            logger.exception("Business rule failed in SQL", extra={"rule": rule.name})
            skipped.append(rule.name)
            continue

        if affected == 0:
            continue

        ratio = affected / source.row_count if source.row_count else 0.0
        findings.append(
            _finding(
                source,
                check_id="business_rule",
                issue=IssueType.BUSINESS_RULE_VIOLATION,
                severity=rule.severity,
                column=rule.columns_used()[0],
                title=f"Business rule failed: {rule.name}",
                explanation=(
                    f"{affected:,} row(s) ({pct(ratio)}) break the rule '{rule.name}'. {sentence}"
                    + (f" {rule.description}" if rule.description else "")
                ),
                affected=affected,
                suffix=rule.name,
            )
        )

    if skipped:
        findings.append(_skipped_rules_finding(source, skipped))
    return findings


def _rule_predicate(rule: object, table: str) -> tuple[str, list[object], str] | None:
    """Return SQL counting the violating rows, its parameters and a sentence.

    Returns None when the rule needs parsing SQL cannot do.
    """

    def counting(predicate: str) -> str:
        return f"SELECT count(*) FROM {table} WHERE {predicate}"  # noqa: S608

    if isinstance(rule, NotNullRule):
        return (
            counting(missing_expression(rule.column)),
            [],
            f"'{rule.column}' is mandatory but these rows leave it empty.",
        )

    if isinstance(rule, RangeRule):
        value = numeric_expression(rule.column)
        clauses = []
        parameters: list[object] = []
        if rule.minimum is not None:
            clauses.append(f"{value} < ?")
            parameters.append(rule.minimum)
        if rule.maximum is not None:
            clauses.append(f"{value} > ?")
            parameters.append(rule.maximum)
        return (
            counting("(" + " OR ".join(clauses) + ")"),
            parameters,
            f"'{rule.column}' must be {rule.describe_bounds()}.",
        )

    if isinstance(rule, AllowedValuesRule):
        column = quote_identifier(rule.column)
        compared = f"trim({column})" if rule.case_sensitive else f"lower(trim({column}))"
        allowed = sorted(rule.allowed_set())
        placeholders = ", ".join("?" for _ in allowed)
        vocabulary = ", ".join(repr(value) for value in rule.values[:8])
        suffix = "…" if len(rule.values) > 8 else ""
        return (
            counting(
                f"NOT {missing_expression(rule.column)} AND {compared} NOT IN ({placeholders})"
            ),
            list(allowed),
            f"'{rule.column}' only accepts: {vocabulary}{suffix}"
            + ("" if rule.case_sensitive else " (case-insensitive)")
            + ".",
        )

    if isinstance(rule, RegexRule):
        column = quote_identifier(rule.column)
        # Anchored at the start, matching Python's re.match rather than re.search.
        return (
            counting(
                f"NOT {missing_expression(rule.column)} AND NOT regexp_matches(trim({column}), ?)"
            ),
            ["^(?:" + rule.pattern + ")"],
            f"'{rule.column}' must match the pattern `{rule.pattern}`.",
        )

    if isinstance(rule, UniqueRule):
        key = ", ".join(normalised_expression(column) for column in rule.columns)
        joined = " + ".join(f"'{column}'" for column in rule.columns)
        # Every group of n identical rows contributes n-1 violations: the first
        # occurrence is the record, the rest are the repeats. Counting the whole group
        # would double-report a pair.
        return (
            f"SELECT coalesce(sum(n - 1), 0) FROM ("  # noqa: S608
            f"SELECT count(*) AS n FROM {table} GROUP BY {key} HAVING count(*) > 1)",
            [],
            f"{joined} must be unique but these rows repeat a value.",
        )

    if isinstance(rule, ComparisonRule) and rule.compare_as == "numeric":
        left, right = numeric_expression(rule.left), numeric_expression(rule.right)
        operator = _OPERATOR_SQL[rule.operator]
        return (
            counting(
                f"{left} IS NOT NULL AND {right} IS NOT NULL AND NOT ({left} {operator} {right})"
            ),
            [],
            f"'{rule.left}' must be {rule.operator} '{rule.right}' (compared as numeric).",
        )

    # Date rules need the seventeen-format parser, which does not exist in SQL.
    return None


def _skipped_rules_finding(source: SqlSource, skipped: list[str]) -> Finding:
    """Say plainly which rules were not evaluated, rather than reporting a clean pass."""
    listed = ", ".join(skipped[:8]) + ("…" if len(skipped) > 8 else "")
    return _finding(
        source,
        check_id="business_rule",
        issue=IssueType.BUSINESS_RULE_VIOLATION,
        severity=Severity.INFO,
        column=None,
        title=f"{len(skipped)} business rule(s) were not evaluated",
        explanation=(
            f"These rules were skipped by the SQL engine: {listed}. A rule is skipped when "
            "it names a column this dataset does not have, or when it needs parsing the "
            "database cannot do - date formats above all. Run the Python engine for those; "
            "a rule that is silently ignored is worse than one that fails."
        ),
        affected=0,
        suffix="sql-skipped",
    )


# ------------------------------------------------------------------ helper


def _finding(
    source: SqlSource,
    *,
    check_id: str,
    issue: IssueType,
    severity: Severity,
    column: str | None,
    title: str,
    explanation: str,
    affected: int,
    details: dict[str, object] | None = None,
    suffix: str | None = None,
) -> Finding:
    """Build a finding, tagging it with the engine that produced it."""
    payload: dict[str, object] = {"engine": "sql"}
    payload.update(details or {})
    return Finding(
        finding_id=make_finding_id(FindingSource.DETERMINISTIC, check_id, issue, suffix or column),
        check_id=check_id,
        issue_type=issue,
        severity=severity,
        column=column,
        title=title,
        explanation=explanation,
        affected_rows=affected,
        row_count=source.row_count,
        row_indices=[],
        details=payload,
    )
