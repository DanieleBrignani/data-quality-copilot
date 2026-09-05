"""Pydantic models for configurable business rules.

Business rules are declared in YAML so a domain expert can add one without touching
Python. Each rule type maps to a small, deterministic predicate; the rule engine turns
violations into ordinary findings.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from dqcopilot.models.enums import Severity


class _RuleBase(BaseModel):
    """Fields shared by every rule type."""

    name: str = Field(min_length=1, max_length=120)
    description: str = ""
    severity: Severity = Severity.HIGH
    enabled: bool = True

    def columns_used(self) -> list[str]:
        """Return the column names this rule needs."""
        raise NotImplementedError


class NotNullRule(_RuleBase):
    """Mandatory column: no missing values allowed."""

    type: Literal["not_null"] = "not_null"
    column: str
    severity: Severity = Severity.CRITICAL

    def columns_used(self) -> list[str]:  # noqa: D102 - inherited
        return [self.column]


class RangeRule(_RuleBase):
    """Numeric column bounded by an inclusive minimum and/or maximum."""

    type: Literal["range"] = "range"
    column: str
    minimum: float | None = None
    maximum: float | None = None

    @model_validator(mode="after")
    def _check_bounds(self) -> RangeRule:
        if self.minimum is None and self.maximum is None:
            raise ValueError(f"Rule '{self.name}': set at least one of minimum/maximum.")
        if self.minimum is not None and self.maximum is not None and self.minimum > self.maximum:
            raise ValueError(f"Rule '{self.name}': minimum is greater than maximum.")
        return self

    def columns_used(self) -> list[str]:  # noqa: D102 - inherited
        return [self.column]

    def describe_bounds(self) -> str:
        """Return a human readable description of the allowed interval."""
        if self.minimum is not None and self.maximum is not None:
            return f"between {self.minimum:g} and {self.maximum:g}"
        if self.minimum is not None:
            return f"greater than or equal to {self.minimum:g}"
        return f"less than or equal to {self.maximum:g}"


class AllowedValuesRule(_RuleBase):
    """Categorical column restricted to a fixed vocabulary."""

    type: Literal["allowed_values"] = "allowed_values"
    column: str
    values: list[str] = Field(min_length=1)
    case_sensitive: bool = False

    def columns_used(self) -> list[str]:  # noqa: D102 - inherited
        return [self.column]

    def normalise(self, value: str) -> str:
        """Normalise a value for comparison against the vocabulary."""
        text = value.strip()
        return text if self.case_sensitive else text.casefold()

    def allowed_set(self) -> set[str]:
        """Return the vocabulary, normalised for comparison."""
        return {self.normalise(value) for value in self.values}


class NoFutureDatesRule(_RuleBase):
    """Date column that must not contain dates after today."""

    type: Literal["no_future_dates"] = "no_future_dates"
    column: str
    #: Days of tolerance, for columns that legitimately hold near-future dates.
    tolerance_days: int = Field(default=0, ge=0)

    def columns_used(self) -> list[str]:  # noqa: D102 - inherited
        return [self.column]


class RegexRule(_RuleBase):
    """Column whose non-empty values must match a regular expression."""

    type: Literal["regex"] = "regex"
    column: str
    pattern: str

    @field_validator("pattern")
    @classmethod
    def _compilable(cls, value: str) -> str:
        import re

        try:
            re.compile(value)
        except re.error as exc:
            raise ValueError(f"Invalid regular expression: {exc}") from exc
        return value

    def columns_used(self) -> list[str]:  # noqa: D102 - inherited
        return [self.column]


class UniqueRule(_RuleBase):
    """Column (or combination of columns) that must not repeat a value."""

    type: Literal["unique"] = "unique"
    columns: list[str] = Field(min_length=1)

    def columns_used(self) -> list[str]:  # noqa: D102 - inherited
        return list(self.columns)


class ComparisonRule(_RuleBase):
    """Relationship between two columns, e.g. ``end_date >= start_date``."""

    type: Literal["comparison"] = "comparison"
    left: str
    operator: Literal["<", "<=", ">", ">=", "==", "!="]
    right: str
    #: How to interpret both columns before comparing.
    compare_as: Literal["numeric", "date"] = "numeric"

    def columns_used(self) -> list[str]:  # noqa: D102 - inherited
        return [self.left, self.right]


BusinessRule = Annotated[
    NotNullRule
    | RangeRule
    | AllowedValuesRule
    | NoFutureDatesRule
    | RegexRule
    | UniqueRule
    | ComparisonRule,
    Field(discriminator="type"),
]


class BusinessRuleSet(BaseModel):
    """A named collection of business rules loaded from YAML."""

    version: int = 1
    name: str = "default"
    description: str = ""
    rules: list[BusinessRule] = Field(default_factory=list)

    @model_validator(mode="after")
    def _unique_names(self) -> BusinessRuleSet:
        seen: set[str] = set()
        for rule in self.rules:
            if rule.name in seen:
                raise ValueError(f"Duplicate rule name: '{rule.name}'.")
            seen.add(rule.name)
        return self

    def enabled_rules(self) -> list[BusinessRule]:
        """Return only the rules that are switched on."""
        return [rule for rule in self.rules if rule.enabled]

    def applicable(self, columns: set[str]) -> tuple[list[BusinessRule], list[BusinessRule]]:
        """Split the enabled rules into (applicable, skipped).

        A rule is skipped when the dataset does not contain every column it needs.
        Skipping rather than failing is what lets one rule file serve several datasets.
        """
        applicable: list[BusinessRule] = []
        skipped: list[BusinessRule] = []
        for rule in self.enabled_rules():
            if set(rule.columns_used()).issubset(columns):
                applicable.append(rule)
            else:
                skipped.append(rule)
        return applicable, skipped
