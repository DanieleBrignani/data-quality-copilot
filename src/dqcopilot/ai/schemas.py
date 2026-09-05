"""Pydantic schemas for every Anthropic response.

These models are passed to ``client.messages.parse(output_format=...)``, so the API
constrains generation to the schema *and* the SDK validates the result. Validation is
still treated as untrusted input afterwards: see :mod:`dqcopilot.ai.grounding`, which
checks that whatever the model returned actually refers to this dataset.

Every field is deliberately concrete - no free-form dictionaries - so a malformed or
adversarial response fails validation instead of flowing into the application.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

MAX_ITEMS = 25


class _StrictModel(BaseModel):
    """Base for every AI schema.

    ``extra="forbid"`` matters: without it, a response carrying unexpected fields
    validates against a model whose own fields all have defaults, and a completely
    wrong payload would be silently accepted as an empty result.
    """

    model_config = ConfigDict(extra="forbid")


class ColumnInterpretation(_StrictModel):
    """What the model thinks a column means."""

    column: str = Field(max_length=255, description="Exact column name from the input.")
    meaning: str = Field(
        max_length=400, description="One sentence describing what this column holds."
    )
    likely_unit: str = Field(
        default="",
        max_length=80,
        description="Unit or currency if apparent, otherwise an empty string.",
    )
    is_personal_data: bool = Field(
        description="True if this column probably holds personal data about individuals."
    )
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str = Field(max_length=400)


class ColumnInterpretationResponse(_StrictModel):
    """Top-level response for the column interpretation task."""

    interpretations: list[ColumnInterpretation] = Field(default_factory=list, max_length=MAX_ITEMS)


class CategoryMappingSuggestion(_StrictModel):
    """A proposal to treat two category values as the same thing."""

    column: str = Field(max_length=255)
    from_value: str = Field(
        max_length=200, description="The non-standard value, exactly as it appears."
    )
    to_value: str = Field(
        max_length=200, description="The value it should become, exactly as it appears."
    )
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str = Field(max_length=300)


class CategoryMappingResponse(_StrictModel):
    """Top-level response for the category mapping task."""

    mappings: list[CategoryMappingSuggestion] = Field(default_factory=list, max_length=MAX_ITEMS)


class AnomalyExplanation(_StrictModel):
    """A plain-language explanation of one detected problem."""

    finding_reference: str = Field(
        max_length=64, description="The finding id that was supplied in the prompt."
    )
    plain_language: str = Field(
        max_length=600, description="What the problem means for someone using this data."
    )
    likely_cause: str = Field(max_length=400)
    suggested_action: str = Field(max_length=400)
    business_impact: str = Field(
        default="",
        max_length=300,
        description="What could go wrong downstream if this is not fixed.",
    )


class AnomalyExplanationResponse(_StrictModel):
    """Top-level response for the anomaly explanation task."""

    explanations: list[AnomalyExplanation] = Field(default_factory=list, max_length=MAX_ITEMS)


class SuggestedRule(_StrictModel):
    """A semantic data quality rule the model thinks this dataset should have."""

    name: str = Field(max_length=120, description="Short snake_case identifier.")
    rule_type: str = Field(
        max_length=32,
        description="One of: not_null, range, allowed_values, no_future_dates, regex, unique.",
    )
    column: str = Field(max_length=255)
    description: str = Field(max_length=400, description="Why this rule matters.")
    minimum: float | None = Field(default=None, description="For range rules only.")
    maximum: float | None = Field(default=None, description="For range rules only.")
    allowed_values: list[str] = Field(
        default_factory=list,
        max_length=40,
        description="For allowed_values rules only.",
    )
    confidence: float = Field(ge=0.0, le=1.0)


class SuggestedRuleResponse(_StrictModel):
    """Top-level response for the rule suggestion task."""

    rules: list[SuggestedRule] = Field(default_factory=list, max_length=MAX_ITEMS)


#: Rule types the application knows how to turn into a real business rule.
SUPPORTED_RULE_TYPES: frozenset[str] = frozenset(
    {"not_null", "range", "allowed_values", "no_future_dates", "regex", "unique"}
)
