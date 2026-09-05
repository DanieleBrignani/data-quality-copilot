"""Checks for values that are the wrong shape: bad dates, bad emails, bad numbers."""

from __future__ import annotations

from datetime import date

import pandas as pd

from dqcopilot.models.enums import FindingSource, IssueType, SemanticType, Severity
from dqcopilot.models.findings import Finding, make_finding_id
from dqcopilot.profiling.type_inference import (
    EMAIL_RE,
    analyze_dates,
    coerce_numeric,
    format_label,
    missing_mask,
    parse_dates_with_format,
    to_clean_strings,
)
from dqcopilot.validation.base import (
    Check,
    CheckContext,
    name_matches,
    pct,
    ratio_severity,
    sample_indices,
)
from dqcopilot.validation.registry import register_check

#: Values further than this many standard deviations from the mean are called suspicious.
OUTLIER_SIGMA = 5.0
#: A column needs at least this many numeric values before outlier detection is meaningful.
MIN_ROWS_FOR_OUTLIERS = 20


def _mask_for(series: pd.Series, predicate_mask: pd.Series) -> pd.Series:
    """Align a mask computed on a subset back onto the full column index."""
    full = pd.Series(False, index=series.index)
    full.loc[predicate_mask.index] = predicate_mask.to_numpy()
    return full


@register_check
class InvalidDateCheck(Check):
    """Report values in a date column that are not real calendar dates."""

    check_id = "invalid_date"
    title = "Invalid dates"
    description = "Finds values in date columns that no known date format can parse."

    def run(self, context: CheckContext) -> list[Finding]:  # noqa: D102 - inherited
        findings: list[Finding] = []
        for column in context.columns():
            if not column.semantic_type.is_temporal:
                continue
            series = context.series(column.name)
            if pd.api.types.is_datetime64_any_dtype(series):
                continue  # Excel already validated these.

            analysis = analyze_dates(series[~missing_mask(series)])
            if not analysis.unparseable_indices:
                continue

            affected = len(analysis.unparseable_indices)
            ratio = affected / context.row_count if context.row_count else 0.0
            examples = [
                str(value) for value in series.loc[analysis.unparseable_indices[:5]].tolist()
            ]

            findings.append(
                Finding(
                    finding_id=make_finding_id(
                        FindingSource.DETERMINISTIC,
                        self.check_id,
                        IssueType.INVALID_DATE,
                        column.name,
                    ),
                    check_id=self.check_id,
                    issue_type=IssueType.INVALID_DATE,
                    severity=ratio_severity(ratio),
                    column=column.name,
                    title=f"Invalid dates in '{column.name}'",
                    explanation=(
                        f"{affected:,} value(s) in '{column.name}' ({pct(ratio)}) are not real "
                        "calendar dates - either the format is unrecognised or the day does "
                        f"not exist (for example 30 February). Examples: "
                        f"{', '.join(repr(value) for value in examples)}."
                    ),
                    affected_rows=affected,
                    row_count=context.row_count,
                    row_indices=analysis.unparseable_indices,
                    details={"examples": examples},
                )
            )
        return findings


@register_check
class InconsistentDateFormatCheck(Check):
    """Report date columns written in more than one format."""

    check_id = "inconsistent_date_format"
    title = "Inconsistent date formats"
    description = "Finds date columns where no single format explains every value."

    def run(self, context: CheckContext) -> list[Finding]:  # noqa: D102 - inherited
        findings: list[Finding] = []
        for column in context.columns():
            if not column.semantic_type.is_temporal:
                continue
            series = context.series(column.name)
            if pd.api.types.is_datetime64_any_dtype(series):
                continue

            analysis = analyze_dates(series[~missing_mask(series)])
            if analysis.parsed_count == 0 or analysis.is_format_consistent:
                continue

            observed = sorted(analysis.format_counts.items(), key=lambda item: -item[1])
            readable = ", ".join(
                f"{format_label(fmt)} ({count:,} values)" for fmt, count in observed[:4]
            )

            findings.append(
                Finding(
                    finding_id=make_finding_id(
                        FindingSource.DETERMINISTIC,
                        self.check_id,
                        IssueType.INCONSISTENT_DATE_FORMAT,
                        column.name,
                    ),
                    check_id=self.check_id,
                    issue_type=IssueType.INCONSISTENT_DATE_FORMAT,
                    severity=Severity.HIGH,
                    column=column.name,
                    title=f"Mixed date formats in '{column.name}'",
                    explanation=(
                        f"No single date format explains every value in '{column.name}'. "
                        f"Formats observed: {readable}. Mixed formats are dangerous because "
                        "01/02/2024 means two different days depending on the convention; "
                        "a human must confirm which one applies before parsing."
                    ),
                    affected_rows=analysis.parsed_count,
                    row_count=context.row_count,
                    details={
                        "format_counts": {format_label(fmt): count for fmt, count in observed[:10]}
                    },
                    confidence=0.9,
                )
            )
        return findings


@register_check
class InvalidEmailCheck(Check):
    """Report malformed email addresses."""

    check_id = "invalid_email"
    title = "Invalid email addresses"
    description = "Finds values in email columns that are not valid addresses."

    def run(self, context: CheckContext) -> list[Finding]:  # noqa: D102 - inherited
        findings: list[Finding] = []
        for column in context.columns():
            if column.semantic_type is not SemanticType.EMAIL:
                continue

            series = context.series(column.name)
            present = to_clean_strings(series)[~missing_mask(series)]
            if present.empty:
                continue

            invalid = ~present.str.strip().map(lambda value: bool(EMAIL_RE.match(value)))
            invalid = invalid.fillna(True).astype(bool)
            affected = int(invalid.sum())
            if affected == 0:
                continue

            mask = _mask_for(series, invalid)
            ratio = affected / context.row_count if context.row_count else 0.0
            examples = present[invalid].head(5).tolist()

            findings.append(
                Finding(
                    finding_id=make_finding_id(
                        FindingSource.DETERMINISTIC,
                        self.check_id,
                        IssueType.INVALID_EMAIL,
                        column.name,
                    ),
                    check_id=self.check_id,
                    issue_type=IssueType.INVALID_EMAIL,
                    severity=ratio_severity(ratio),
                    column=column.name,
                    title=f"Invalid email addresses in '{column.name}'",
                    explanation=(
                        f"{affected:,} value(s) in '{column.name}' ({pct(ratio)}) are not "
                        "valid email addresses - a missing '@', a missing domain, a trailing "
                        "dot or stray characters. These records cannot be contacted."
                    ),
                    affected_rows=affected,
                    row_count=context.row_count,
                    row_indices=sample_indices(mask),
                    details={"examples": [str(value) for value in examples]},
                )
            )
        return findings


@register_check
class NumericStoredAsTextCheck(Check):
    """Report numeric columns whose text form blocks a plain numeric cast.

    Every column of a CSV is technically text, so "this column is a string" would fire
    on every numeric column of every CSV and mean nothing. The check therefore looks for
    values that a naive ``float()`` would *reject*: currency symbols, thousands
    separators, percent signs, accounting parentheses or words. A column of clean
    ``"123"`` strings is left alone, because reading it as a number is unambiguous.
    """

    check_id = "numeric_stored_as_text"
    title = "Numbers stored as text"
    description = "Finds numeric columns decorated with currency symbols or separators."

    #: Values matching this are castable with a plain ``float()`` and need no fix.
    PLAIN_NUMBER = r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?"

    def run(self, context: CheckContext) -> list[Finding]:  # noqa: D102 - inherited
        findings: list[Finding] = []
        for column in context.columns():
            series = context.series(column.name)
            if not column.semantic_type.is_numeric:
                continue
            if pd.api.types.is_numeric_dtype(series):
                continue  # already a real number

            present = series[~missing_mask(series)]
            if present.empty:
                continue

            numeric = coerce_numeric(present)
            parseable = int(numeric.notna().sum())
            unparseable = int(numeric.isna().sum())

            text = to_clean_strings(present)
            plain = text.str.strip().str.fullmatch(self.PLAIN_NUMBER).fillna(False).astype(bool)
            decorated = text[numeric.notna() & ~plain]
            if decorated.empty and unparseable == 0:
                continue  # clean numeric strings: nothing to fix

            findings.append(
                Finding(
                    finding_id=make_finding_id(
                        FindingSource.DETERMINISTIC,
                        self.check_id,
                        IssueType.NUMERIC_STORED_AS_TEXT,
                        column.name,
                    ),
                    check_id=self.check_id,
                    issue_type=IssueType.NUMERIC_STORED_AS_TEXT,
                    severity=Severity.MEDIUM,
                    column=column.name,
                    title=f"'{column.name}' holds numbers as text",
                    explanation=(
                        f"'{column.name}' is numeric but {len(decorated):,} value(s) carry "
                        "formatting that a plain numeric conversion would reject - currency "
                        "symbols, thousands separators, percent signs or parentheses"
                        + (
                            f", and {unparseable:,} value(s) are not numbers at all"
                            if unparseable
                            else ""
                        )
                        + ". Left as text the column cannot be summed or averaged, and it "
                        "sorts alphabetically ('100' before '9'). Examples: "
                        f"{', '.join(repr(str(v)) for v in decorated.head(4).tolist())}."
                    ),
                    affected_rows=parseable,
                    row_count=context.row_count,
                    details={
                        "parseable": parseable,
                        "unparseable": unparseable,
                        "decorated": len(decorated),
                        "examples": [str(value) for value in decorated.head(5).tolist()],
                    },
                )
            )
        return findings


@register_check
class ImpossibleNumericCheck(Check):
    """Report values that a column's own meaning makes impossible.

    Only two universally safe rules are applied here, because anything more depends on
    the business: quantities/amounts implied by the column name must not be negative,
    and ages must fall in a human range. Everything else belongs in a business rule.
    """

    check_id = "impossible_numeric"
    title = "Impossible numeric values"
    description = "Finds negative amounts and out-of-range ages."

    NON_NEGATIVE_HINTS = (
        "revenue",
        "amount",
        "price",
        "cost",
        "total",
        "quantity",
        "qty",
        "stock",
        "salary",
        "sales",
        "turnover",
        "budget",
        "fee",
        "spend",
    )
    AGE_HINTS = ("age",)
    AGE_MIN = 0.0
    AGE_MAX = 120.0

    def run(self, context: CheckContext) -> list[Finding]:  # noqa: D102 - inherited
        findings: list[Finding] = []
        for column in context.columns():
            if not column.semantic_type.is_numeric:
                continue
            numeric = coerce_numeric(context.series(column.name))

            if name_matches(column.name, self.AGE_HINTS):
                mask = (numeric < self.AGE_MIN) | (numeric > self.AGE_MAX)
                reason = f"outside the plausible human range {self.AGE_MIN:g}-{self.AGE_MAX:g}"
            elif name_matches(column.name, self.NON_NEGATIVE_HINTS):
                mask = numeric < 0
                reason = "negative, which this kind of amount cannot be"
            else:
                continue

            mask = mask.fillna(False).astype(bool)
            affected = int(mask.sum())
            if affected == 0:
                continue

            ratio = affected / context.row_count if context.row_count else 0.0
            examples = numeric[mask].head(5).tolist()

            findings.append(
                Finding(
                    finding_id=make_finding_id(
                        FindingSource.DETERMINISTIC,
                        self.check_id,
                        IssueType.IMPOSSIBLE_NUMERIC,
                        column.name,
                    ),
                    check_id=self.check_id,
                    issue_type=IssueType.IMPOSSIBLE_NUMERIC,
                    severity=Severity.HIGH,
                    column=column.name,
                    title=f"Impossible values in '{column.name}'",
                    explanation=(
                        f"{affected:,} value(s) in '{column.name}' ({pct(ratio)}) are {reason}. "
                        f"Examples: {', '.join(f'{value:g}' for value in examples)}. This "
                        "usually means a sign error, a failed import or a placeholder value."
                    ),
                    affected_rows=affected,
                    row_count=context.row_count,
                    row_indices=sample_indices(mask),
                    details={"examples": [float(value) for value in examples], "reason": reason},
                )
            )
        return findings


@register_check
class SuspiciousNumericCheck(Check):
    """Report extreme numeric outliers, flagged for review rather than correction."""

    check_id = "suspicious_numeric"
    title = "Suspicious numeric outliers"
    description = "Finds values many standard deviations away from the column mean."

    def run(self, context: CheckContext) -> list[Finding]:  # noqa: D102 - inherited
        findings: list[Finding] = []
        for column in context.columns():
            if not column.semantic_type.is_numeric:
                continue
            numeric = coerce_numeric(context.series(column.name)).dropna()
            if numeric.size < MIN_ROWS_FOR_OUTLIERS:
                continue

            mean = float(numeric.mean())
            std = float(numeric.std())
            if std <= 0:
                continue

            deviation = (numeric - mean).abs() / std
            outliers = deviation > OUTLIER_SIGMA
            affected = int(outliers.sum())
            if affected == 0:
                continue

            mask = _mask_for(context.series(column.name), outliers)
            examples = numeric[outliers].head(5).tolist()

            findings.append(
                Finding(
                    finding_id=make_finding_id(
                        FindingSource.DETERMINISTIC,
                        self.check_id,
                        IssueType.SUSPICIOUS_NUMERIC,
                        column.name,
                    ),
                    check_id=self.check_id,
                    issue_type=IssueType.SUSPICIOUS_NUMERIC,
                    severity=Severity.LOW,
                    column=column.name,
                    title=f"Extreme values in '{column.name}'",
                    explanation=(
                        f"{affected:,} value(s) in '{column.name}' sit more than "
                        f"{OUTLIER_SIGMA:g} standard deviations from the mean "
                        f"({mean:,.2f} ± {std:,.2f}). Examples: "
                        f"{', '.join(f'{value:,.2f}' for value in examples)}. An outlier is "
                        "not necessarily wrong - this is flagged for review, never corrected."
                    ),
                    affected_rows=affected,
                    row_count=context.row_count,
                    row_indices=sample_indices(mask),
                    details={
                        "mean": round(mean, 4),
                        "std_dev": round(std, 4),
                        "sigma_threshold": OUTLIER_SIGMA,
                        "examples": [float(value) for value in examples],
                    },
                    confidence=0.6,
                )
            )
        return findings


@register_check
class FutureDateCheck(Check):
    """Report dates in the future in columns that describe past events."""

    check_id = "future_date"
    title = "Dates in the future"
    description = "Finds future dates in columns that record something that already happened."

    PAST_EVENT_HINTS = (
        "birth",
        "born",
        "dob",
        "signup",
        "registered",
        "registration",
        "created",
        "order",
        "purchase",
        "invoice",
        "payment",
        "hire",
        "hired",
        "joined",
        "onboarded",
        "onboarding",
        "opened",
        "issued",
    )

    def run(self, context: CheckContext) -> list[Finding]:  # noqa: D102 - inherited
        findings: list[Finding] = []
        today = pd.Timestamp(date.today())

        for column in context.columns():
            if not column.semantic_type.is_temporal:
                continue
            if not name_matches(column.name, self.PAST_EVENT_HINTS):
                continue

            parsed = parse_dates_with_format(context.series(column.name), None)
            mask = (parsed > today).fillna(False).astype(bool)
            affected = int(mask.sum())
            if affected == 0:
                continue

            ratio = affected / context.row_count if context.row_count else 0.0
            examples = [str(value.date()) for value in parsed[mask].head(5)]

            findings.append(
                Finding(
                    finding_id=make_finding_id(
                        FindingSource.DETERMINISTIC,
                        self.check_id,
                        IssueType.FUTURE_DATE,
                        column.name,
                    ),
                    check_id=self.check_id,
                    issue_type=IssueType.FUTURE_DATE,
                    severity=Severity.HIGH,
                    column=column.name,
                    title=f"Future dates in '{column.name}'",
                    explanation=(
                        f"{affected:,} value(s) in '{column.name}' ({pct(ratio)}) are later "
                        f"than today ({today.date()}). The column name suggests it records "
                        "something that has already happened, so a future date points to a "
                        f"typo or a wrong century. Examples: {', '.join(examples)}."
                    ),
                    affected_rows=affected,
                    row_count=context.row_count,
                    row_indices=sample_indices(mask),
                    details={"examples": examples, "evaluated_on": str(today.date())},
                )
            )
        return findings
