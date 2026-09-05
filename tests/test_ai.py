"""AI tests. No test in this file makes a network call.

The Anthropic SDK is replaced by a stub whose ``messages.parse`` returns whatever the
test tells it to - including malformed and adversarial responses, which is the point:
the guarantees worth testing are the ones that hold when the model misbehaves.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
import pytest
from pydantic import BaseModel, ValidationError

from dqcopilot.ai import (
    AiSuggester,
    AnthropicClient,
    CategoryMappingResponse,
    CategoryMappingSuggestion,
    ColumnInterpretation,
    ColumnInterpretationResponse,
    SuggestedRule,
    SuggestedRuleResponse,
    ground_category_mappings,
    ground_column_interpretations,
    ground_suggested_rules,
    mask_value,
    should_mask,
)
from dqcopilot.ai.payload import category_payload, dataset_payload
from dqcopilot.ai.schemas import AnomalyExplanationResponse
from dqcopilot.config import Settings
from dqcopilot.models import FindingSource, SemanticType
from dqcopilot.profiling import profile_dataset
from dqcopilot.services import analyze_bytes

# --------------------------------------------------------------------------- stubs


@dataclass
class _Usage:
    input_tokens: int = 500
    output_tokens: int = 120


class _Response:
    def __init__(self, parsed: Any, usage: _Usage | None = None) -> None:
        self.parsed_output = parsed
        self.usage = usage or _Usage()


class _Messages:
    """Dispatches on ``output_format`` rather than call order.

    The suggester makes a variable number of calls (one per candidate category
    column, and it skips tasks that have nothing to ask about), so an
    order-based stub would silently hand the wrong payload to the wrong task.
    """

    def __init__(
        self,
        payloads: dict[type, Any],
        errors: dict[type, Exception] | None = None,
    ) -> None:
        self._payloads = payloads
        self._errors = errors or {}
        self.calls: list[dict[str, Any]] = []

    def parse(self, **kwargs: Any) -> _Response:
        self.calls.append(kwargs)
        schema = kwargs.get("output_format")

        if schema in self._errors:
            raise self._errors[schema]
        if schema in self._payloads:
            return _Response(self._payloads[schema])
        if isinstance(schema, type) and issubclass(schema, BaseModel):
            return _Response(schema())  # an empty, valid response
        return _Response(None)


class StubAnthropic:
    """Stand-in for ``anthropic.Anthropic``.

    Pass response instances (matched by their type) and optionally a mapping of
    schema type to the exception that call should raise.
    """

    def __init__(
        self,
        *payloads: Any,
        errors: dict[type, Exception] | None = None,
    ) -> None:
        self.messages = _Messages({type(item): item for item in payloads}, errors)


class SingleShotStub:
    """Returns one scripted outcome for any call, whatever the schema.

    Used by the client-level tests, which are about how the wrapper reacts to a bad
    response rather than about which task asked for it.
    """

    class _Single:
        def __init__(self, outcome: Any) -> None:
            self._outcome = outcome
            self.calls: list[dict[str, Any]] = []

        def parse(self, **kwargs: Any) -> _Response:
            self.calls.append(kwargs)
            if isinstance(self._outcome, Exception):
                raise self._outcome
            return _Response(self._outcome)

    def __init__(self, outcome: Any) -> None:
        self.messages = SingleShotStub._Single(outcome)


@pytest.fixture
def ai_settings() -> Settings:
    return Settings(
        anthropic_api_key="sk-ant-test-not-a-real-key",  # noqa: S106 - stub only
        anthropic_model="claude-opus-5",
        persistence_enabled=False,
    )


@pytest.fixture
def messy_csv() -> bytes:
    frame = pd.DataFrame(
        {
            "customer_id": [f"CUST-{i:04d}" for i in range(12)],
            "full_name": [f"Person {i}" for i in range(12)],
            "email": [f"person{i}@example.com" for i in range(12)],
            "country": ["FR", "France", "FR", "DE", "Germany", "DE"] * 2,
            "revenue": ["1,200.50", "900", "-50", "1,000", "2,500", "700"] * 2,
            "signup_date": ["2024-01-15"] * 12,
        }
    )
    return frame.to_csv(index=False).encode()


# --------------------------------------------------------------------------- payload


class TestMasking:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("alice.martin@example.com", "aaaaa.aaaaaa@aaaaaaa.aaa"),
            ("CUST-00042", "aaaa-99999"),
            ("FR12345678", "aa99999999"),
            ("", ""),
        ],
    )
    def test_mask_preserves_shape_not_content(self, raw: str, expected: str) -> None:
        assert mask_value(raw) == expected

    def test_mask_truncates_long_values(self) -> None:
        masked = mask_value("x" * 200)
        assert len(masked) <= 41
        assert masked.endswith("…")

    def test_email_and_identifier_columns_are_masked(self, messy_csv: bytes) -> None:
        frame = pd.read_csv(pd.io.common.BytesIO(messy_csv), dtype=str)
        profile = profile_dataset(frame, "t")
        for name in ("email", "customer_id", "full_name"):
            column = profile.column(name)
            assert column is not None
            assert should_mask(column), f"{name} should be masked"

    def test_ordinary_categorical_is_not_masked(self, messy_csv: bytes) -> None:
        frame = pd.read_csv(pd.io.common.BytesIO(messy_csv), dtype=str)
        profile = profile_dataset(frame, "t")
        column = profile.column("country")
        assert column is not None
        assert not should_mask(column)


class TestPayload:
    def test_never_contains_full_dataset(self, messy_csv: bytes, ai_settings: Settings) -> None:
        result = analyze_bytes(messy_csv, "c.csv", settings=ai_settings)
        payload = dataset_payload(result.profile, result.frame, ai_settings)
        rendered = str(payload)

        # Real email addresses and identifiers must not appear.
        assert "person0@example.com" not in rendered
        assert "CUST-0000" not in rendered
        # Row count is metadata, and must be present.
        assert payload["row_count"] == 12

    def test_respects_the_column_cap(self, messy_csv: bytes) -> None:
        settings = Settings(anthropic_api_key=None, ai_max_columns=2)
        result = analyze_bytes(messy_csv, "c.csv", settings=settings)
        payload = dataset_payload(result.profile, result.frame, settings)
        assert len(payload["columns"]) == 2

    def test_samples_can_be_switched_off_entirely(self, messy_csv: bytes) -> None:
        settings = Settings(anthropic_api_key=None, ai_send_samples=False)
        result = analyze_bytes(messy_csv, "c.csv", settings=settings)
        payload = dataset_payload(result.profile, result.frame, settings)
        assert all("examples" not in column for column in payload["columns"])

    def test_category_payload_sends_levels_and_counts(
        self, messy_csv: bytes, ai_settings: Settings
    ) -> None:
        result = analyze_bytes(messy_csv, "c.csv", settings=ai_settings)
        column = result.profile.column("country")
        assert column is not None
        payload = category_payload(column, result.frame["country"], ai_settings)
        values = {level["value"] for level in payload["levels"]}
        assert {"FR", "France", "DE", "Germany"} <= values


# --------------------------------------------------------------------------- schemas


class TestSchemaValidation:
    def test_valid_response_parses(self) -> None:
        response = CategoryMappingResponse.model_validate(
            {
                "mappings": [
                    {
                        "column": "country",
                        "from_value": "France",
                        "to_value": "FR",
                        "confidence": 0.9,
                        "reasoning": "ISO code",
                    }
                ]
            }
        )
        assert len(response.mappings) == 1

    def test_confidence_out_of_range_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            CategoryMappingSuggestion(
                column="c", from_value="a", to_value="b", confidence=1.5, reasoning="x"
            )

    def test_missing_required_field_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            CategoryMappingSuggestion(column="c", from_value="a", to_value="b")  # type: ignore[call-arg]

    def test_oversized_list_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            CategoryMappingResponse(
                mappings=[
                    CategoryMappingSuggestion(
                        column="c",
                        from_value=f"a{i}",
                        to_value="b",
                        confidence=0.9,
                        reasoning="x",
                    )
                    for i in range(40)
                ]
            )

    def test_overlong_string_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ColumnInterpretation(
                column="c",
                meaning="x" * 500,
                is_personal_data=False,
                confidence=0.9,
                reasoning="y",
            )


# --------------------------------------------------------------------------- grounding


class TestGroundingCategoryMappings:
    @pytest.fixture
    def frame(self) -> pd.DataFrame:
        return pd.DataFrame({"country": ["FR", "France", "DE", "FR", "Germany"]})

    def _suggestion(self, **kwargs: Any) -> CategoryMappingSuggestion:
        defaults = {
            "column": "country",
            "from_value": "France",
            "to_value": "FR",
            "confidence": 0.9,
            "reasoning": "same country",
        }
        return CategoryMappingSuggestion(**{**defaults, **kwargs})

    def test_accepts_a_grounded_mapping(self, frame: pd.DataFrame) -> None:
        report = ground_category_mappings([self._suggestion()], frame)
        assert len(report.accepted) == 1
        assert not report.rejected

    def test_rejects_unknown_column(self, frame: pd.DataFrame) -> None:
        report = ground_category_mappings([self._suggestion(column="nope")], frame)
        assert not report.accepted
        assert "not in the dataset" in report.rejection_reasons[0]

    def test_rejects_source_value_that_does_not_exist(self, frame: pd.DataFrame) -> None:
        """The model hallucinating a value is the failure mode this exists to stop."""
        report = ground_category_mappings([self._suggestion(from_value="Francia")], frame)
        assert not report.accepted
        assert "does not occur" in report.rejection_reasons[0]

    def test_rejects_target_value_that_would_be_invented(self, frame: pd.DataFrame) -> None:
        report = ground_category_mappings([self._suggestion(to_value="FRA")], frame)
        assert not report.accepted
        assert "invent" in report.rejection_reasons[0]

    def test_rejects_low_confidence(self, frame: pd.DataFrame) -> None:
        report = ground_category_mappings([self._suggestion(confidence=0.2)], frame)
        assert not report.accepted
        assert "below" in report.rejection_reasons[0]

    def test_rejects_self_mapping(self, frame: pd.DataFrame) -> None:
        report = ground_category_mappings([self._suggestion(to_value="France")], frame)
        assert not report.accepted

    def test_rejects_contradictory_mappings(self, frame: pd.DataFrame) -> None:
        report = ground_category_mappings(
            [self._suggestion(), self._suggestion(to_value="DE")], frame
        )
        assert len(report.accepted) == 1
        assert len(report.rejected) == 1

    def test_rejects_cycles(self, frame: pd.DataFrame) -> None:
        report = ground_category_mappings(
            [
                self._suggestion(from_value="France", to_value="FR"),
                self._suggestion(from_value="FR", to_value="France"),
            ],
            frame,
        )
        assert len(report.accepted) == 1
        assert "map to each other" in report.rejection_reasons[0]


class TestGroundingRules:
    @pytest.fixture
    def frame(self) -> pd.DataFrame:
        return pd.DataFrame({"revenue": ["100", "200"], "status": ["open", "closed"]})

    def _rule(self, **kwargs: Any) -> SuggestedRule:
        defaults = {
            "name": "revenue_positive",
            "rule_type": "range",
            "column": "revenue",
            "description": "Revenue cannot be negative.",
            "minimum": 0.0,
            "confidence": 0.9,
        }
        return SuggestedRule(**{**defaults, **kwargs})

    def test_accepts_a_valid_rule(self, frame: pd.DataFrame) -> None:
        assert len(ground_suggested_rules([self._rule()], frame).accepted) == 1

    def test_rejects_unsupported_rule_type(self, frame: pd.DataFrame) -> None:
        report = ground_suggested_rules([self._rule(rule_type="telepathy")], frame)
        assert not report.accepted
        assert "not supported" in report.rejection_reasons[0]

    def test_rejects_unknown_column(self, frame: pd.DataFrame) -> None:
        assert not ground_suggested_rules([self._rule(column="nope")], frame).accepted

    def test_rejects_inverted_range(self, frame: pd.DataFrame) -> None:
        report = ground_suggested_rules([self._rule(minimum=100.0, maximum=1.0)], frame)
        assert "inverted" in report.rejection_reasons[0]

    def test_rejects_range_without_bounds(self, frame: pd.DataFrame) -> None:
        report = ground_suggested_rules([self._rule(minimum=None, maximum=None)], frame)
        assert "at least one bound" in report.rejection_reasons[0]

    def test_rejects_vocabulary_disjoint_from_the_data(self, frame: pd.DataFrame) -> None:
        report = ground_suggested_rules(
            [
                self._rule(
                    name="status_known",
                    rule_type="allowed_values",
                    column="status",
                    allowed_values=["alpha", "beta"],
                    minimum=None,
                )
            ],
            frame,
        )
        assert "None of the proposed values" in report.rejection_reasons[0]

    def test_accepts_vocabulary_that_matches_the_data(self, frame: pd.DataFrame) -> None:
        report = ground_suggested_rules(
            [
                self._rule(
                    name="status_known",
                    rule_type="allowed_values",
                    column="status",
                    allowed_values=["open", "closed", "pending"],
                    minimum=None,
                )
            ],
            frame,
        )
        assert len(report.accepted) == 1


class TestGroundingInterpretations:
    def test_rejects_unknown_and_duplicate_columns(self, messy_csv: bytes) -> None:
        result = analyze_bytes(messy_csv, "c.csv", settings=Settings(anthropic_api_key=None))

        def interpretation(column: str) -> ColumnInterpretation:
            return ColumnInterpretation(
                column=column,
                meaning="something",
                is_personal_data=False,
                confidence=0.9,
                reasoning="because",
            )

        report = ground_column_interpretations(
            [interpretation("country"), interpretation("country"), interpretation("ghost")],
            result.profile,
        )
        assert len(report.accepted) == 1
        assert len(report.rejected) == 2


# --------------------------------------------------------------------------- client


class TestAnthropicClient:
    def test_reduced_mode_without_a_key(self) -> None:
        client = AnthropicClient(Settings(anthropic_api_key=None))
        assert not client.enabled

        result = client.parse(
            task="t", system="s", user_content="{}", output_format=CategoryMappingResponse
        )
        assert not result.ok
        assert "ANTHROPIC_API_KEY" in result.error

    def test_successful_call_records_usage(self, ai_settings: Settings) -> None:
        payload = CategoryMappingResponse(
            mappings=[
                CategoryMappingSuggestion(
                    column="country",
                    from_value="France",
                    to_value="FR",
                    confidence=0.9,
                    reasoning="iso",
                )
            ]
        )
        client = AnthropicClient(ai_settings, client=StubAnthropic(payload))
        result = client.parse(
            task="t", system="s", user_content="{}", output_format=CategoryMappingResponse
        )

        assert result.ok
        assert result.usage.input_tokens == 500
        assert result.usage.output_tokens == 120
        assert result.usage.latency_ms >= 0
        assert client.telemetry.summary()["calls"] == 1

    def test_api_error_is_reported_not_raised(self, ai_settings: Settings) -> None:
        client = AnthropicClient(ai_settings, client=SingleShotStub(RuntimeError("boom")))
        result = client.parse(
            task="t", system="s", user_content="{}", output_format=CategoryMappingResponse
        )

        assert not result.ok
        assert "failed" in result.error.lower()
        assert client.telemetry.failures == 1

    def test_empty_response_is_reported(self, ai_settings: Settings) -> None:
        client = AnthropicClient(ai_settings, client=SingleShotStub(None))
        result = client.parse(
            task="t", system="s", user_content="{}", output_format=CategoryMappingResponse
        )
        assert not result.ok
        assert result.usage.error_type == "empty_response"

    def test_malformed_response_fails_validation(self, ai_settings: Settings) -> None:
        """A response that does not match the schema must not reach the application."""

        class Wrong(BaseModel):
            unexpected: str

        client = AnthropicClient(ai_settings, client=SingleShotStub(Wrong(unexpected="x")))
        result = client.parse(
            task="t", system="s", user_content="{}", output_format=CategoryMappingResponse
        )
        assert not result.ok
        assert result.usage.error_type == "ValidationError"

    def test_error_messages_never_echo_the_request(self, ai_settings: Settings) -> None:
        secret = "sk-ant-secret-value"
        client = AnthropicClient(ai_settings, client=SingleShotStub(RuntimeError(secret)))
        result = client.parse(
            task="t", system="s", user_content=secret, output_format=CategoryMappingResponse
        )
        assert secret not in result.error

    def test_token_ceiling_is_sent(self, ai_settings: Settings) -> None:
        stub = StubAnthropic(CategoryMappingResponse())
        client = AnthropicClient(ai_settings, client=stub)
        client.parse(task="t", system="s", user_content="{}", output_format=CategoryMappingResponse)
        assert stub.messages.calls[0]["max_tokens"] == ai_settings.ai_max_output_tokens
        assert stub.messages.calls[0]["model"] == "claude-opus-5"


# --------------------------------------------------------------------------- suggester


class TestAiSuggester:
    def test_reduced_mode_produces_no_suggestions(self, messy_csv: bytes) -> None:
        settings = Settings(anthropic_api_key=None)
        result = analyze_bytes(messy_csv, "c.csv", settings=settings)
        suggestions = AiSuggester(settings).run(result)

        assert not suggestions.ran
        assert suggestions.is_empty
        assert "ANTHROPIC_API_KEY" in suggestions.errors[0]

    def test_grounded_mapping_becomes_an_approvable_proposal(
        self, messy_csv: bytes, ai_settings: Settings
    ) -> None:
        result = analyze_bytes(messy_csv, "c.csv", settings=ai_settings)
        mapping = CategoryMappingResponse(
            mappings=[
                CategoryMappingSuggestion(
                    column="country",
                    from_value="France",
                    to_value="FR",
                    confidence=0.95,
                    reasoning="ISO code for the same country",
                )
            ]
        )
        stub = StubAnthropic(
            ColumnInterpretationResponse(),
            mapping,
            AnomalyExplanationResponse(),
            SuggestedRuleResponse(),
        )
        suggester = AiSuggester(ai_settings, client=AnthropicClient(ai_settings, client=stub))
        suggestions = suggester.run(result)

        assert suggestions.ran
        proposals = [p for p in suggestions.proposals if p.column == "country"]
        assert len(proposals) == 1
        assert proposals[0].source is FindingSource.AI
        assert proposals[0].requires_extra_care
        assert proposals[0].parameters["mapping"] == {"France": "FR"}

    def test_hallucinated_mapping_is_dropped(self, messy_csv: bytes, ai_settings: Settings) -> None:
        result = analyze_bytes(messy_csv, "c.csv", settings=ai_settings)
        mapping = CategoryMappingResponse(
            mappings=[
                CategoryMappingSuggestion(
                    column="country",
                    from_value="Atlantis",
                    to_value="FR",
                    confidence=0.99,
                    reasoning="confidently wrong",
                )
            ]
        )
        stub = StubAnthropic(
            ColumnInterpretationResponse(),
            mapping,
            AnomalyExplanationResponse(),
            SuggestedRuleResponse(),
        )
        suggester = AiSuggester(ai_settings, client=AnthropicClient(ai_settings, client=stub))
        suggestions = suggester.run(result)

        assert not suggestions.proposals
        assert any("does not occur" in reason for reason in suggestions.rejected)

    def test_all_ai_findings_are_tagged_and_informational(
        self, messy_csv: bytes, ai_settings: Settings
    ) -> None:
        result = analyze_bytes(messy_csv, "c.csv", settings=ai_settings)
        stub = StubAnthropic(
            ColumnInterpretationResponse(
                interpretations=[
                    ColumnInterpretation(
                        column="revenue",
                        meaning="Money earned per customer",
                        likely_unit="EUR",
                        is_personal_data=False,
                        confidence=0.8,
                        reasoning="name and values",
                    )
                ]
            ),
            CategoryMappingResponse(),
            AnomalyExplanationResponse(),
            SuggestedRuleResponse(),
        )
        suggester = AiSuggester(ai_settings, client=AnthropicClient(ai_settings, client=stub))
        suggestions = suggester.run(result)

        assert suggestions.findings
        for finding in suggestions.findings:
            assert finding.source is FindingSource.AI
            assert finding.is_advisory

    def test_ai_findings_never_change_the_quality_score(
        self, messy_csv: bytes, ai_settings: Settings
    ) -> None:
        from dqcopilot.models.findings import FindingSet
        from dqcopilot.scoring import compute_quality_score

        result = analyze_bytes(messy_csv, "c.csv", settings=ai_settings)
        stub = StubAnthropic(
            ColumnInterpretationResponse(
                interpretations=[
                    ColumnInterpretation(
                        column="revenue",
                        meaning="Money",
                        is_personal_data=False,
                        confidence=0.9,
                        reasoning="x",
                    )
                ]
            ),
            CategoryMappingResponse(),
            AnomalyExplanationResponse(),
            SuggestedRuleResponse(),
        )
        suggester = AiSuggester(ai_settings, client=AnthropicClient(ai_settings, client=stub))
        suggestions = suggester.run(result)

        combined = FindingSet(findings=[*result.findings.findings, *suggestions.findings])
        assert compute_quality_score(result.profile, combined).overall == result.score.overall

    def test_a_failing_task_does_not_lose_the_others(
        self, messy_csv: bytes, ai_settings: Settings
    ) -> None:
        result = analyze_bytes(messy_csv, "c.csv", settings=ai_settings)
        stub = StubAnthropic(
            CategoryMappingResponse(
                mappings=[
                    CategoryMappingSuggestion(
                        column="country",
                        from_value="Germany",
                        to_value="DE",
                        confidence=0.9,
                        reasoning="iso",
                    )
                ]
            ),
            errors={ColumnInterpretationResponse: RuntimeError("first task explodes")},
        )
        suggester = AiSuggester(ai_settings, client=AnthropicClient(ai_settings, client=stub))
        suggestions = suggester.run(result)

        assert suggestions.errors  # the failure was reported
        assert suggestions.proposals  # and the later task still ran

    def test_usage_totals_are_collected(self, messy_csv: bytes, ai_settings: Settings) -> None:
        result = analyze_bytes(messy_csv, "c.csv", settings=ai_settings)
        stub = StubAnthropic(
            ColumnInterpretationResponse(),
            CategoryMappingResponse(),
            AnomalyExplanationResponse(),
            SuggestedRuleResponse(),
        )
        suggester = AiSuggester(ai_settings, client=AnthropicClient(ai_settings, client=stub))
        suggestions = suggester.run(result)

        assert suggestions.usage["calls"] >= 1
        assert suggestions.usage["input_tokens"] > 0

    def test_suggested_rule_is_offered_as_yaml_not_applied(
        self, messy_csv: bytes, ai_settings: Settings
    ) -> None:
        result = analyze_bytes(messy_csv, "c.csv", settings=ai_settings)
        stub = StubAnthropic(
            ColumnInterpretationResponse(),
            CategoryMappingResponse(),
            AnomalyExplanationResponse(),
            SuggestedRuleResponse(
                rules=[
                    SuggestedRule(
                        name="revenue_below_one_million",
                        rule_type="range",
                        column="revenue",
                        description="Revenue cannot be negative.",
                        maximum=1000000.0,
                        confidence=0.9,
                    )
                ]
            ),
        )
        suggester = AiSuggester(ai_settings, client=AnthropicClient(ai_settings, client=stub))
        suggestions = suggester.run(result)

        rule_findings = [f for f in suggestions.findings if f.check_id == "ai_rule_suggestion"]
        assert len(rule_findings) == 1
        assert "maximum: 1e+06" in rule_findings[0].details["yaml"]
        # A suggested rule must never become an applicable correction.
        assert not any(p.finding_id == rule_findings[0].finding_id for p in suggestions.proposals)


class TestNoNetworkAccess:
    def test_the_stub_is_the_only_client_used(self, ai_settings: Settings) -> None:
        """A guard against a real SDK client being constructed by accident."""
        stub = StubAnthropic(CategoryMappingResponse())
        client = AnthropicClient(ai_settings, client=stub)
        client.parse(task="t", system="s", user_content="{}", output_format=CategoryMappingResponse)
        assert len(stub.messages.calls) == 1

    def test_semantic_type_enum_is_stable(self) -> None:
        assert SemanticType.EMAIL.value == "email"
