"""Optional AI suggestions, powered by the Anthropic API.

Everything in this package is advisory. AI output is schema-constrained, validated with
Pydantic, checked against the real dataset, shown separately from deterministic
findings, excluded from the quality score, and applied only after explicit approval.
Without an API key the application runs in reduced mode: this package is simply idle.
"""

from dqcopilot.ai.client import (
    AiResult,
    AiTelemetry,
    AiUnavailableError,
    AiUsage,
    AnthropicClient,
    build_user_content,
)
from dqcopilot.ai.grounding import (
    GroundingReport,
    ground_category_mappings,
    ground_column_interpretations,
    ground_suggested_rules,
)
from dqcopilot.ai.payload import dataset_payload, mask_value, should_mask
from dqcopilot.ai.schemas import (
    AnomalyExplanation,
    AnomalyExplanationResponse,
    CategoryMappingResponse,
    CategoryMappingSuggestion,
    ColumnInterpretation,
    ColumnInterpretationResponse,
    SuggestedRule,
    SuggestedRuleResponse,
)
from dqcopilot.ai.suggester import AiSuggester, AiSuggestions

__all__ = [
    "AiResult",
    "AiSuggester",
    "AiSuggestions",
    "AiTelemetry",
    "AiUnavailableError",
    "AiUsage",
    "AnomalyExplanation",
    "AnomalyExplanationResponse",
    "AnthropicClient",
    "CategoryMappingResponse",
    "CategoryMappingSuggestion",
    "ColumnInterpretation",
    "ColumnInterpretationResponse",
    "GroundingReport",
    "SuggestedRule",
    "SuggestedRuleResponse",
    "build_user_content",
    "dataset_payload",
    "ground_category_mappings",
    "ground_column_interpretations",
    "ground_suggested_rules",
    "mask_value",
    "should_mask",
]
