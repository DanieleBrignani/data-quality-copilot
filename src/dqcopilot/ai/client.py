"""Anthropic API client wrapper.

Responsibilities kept deliberately narrow: build a request, enforce the token ceiling
and timeout, validate the response against a Pydantic schema, and record latency and
token usage. It never decides what to send (see :mod:`dqcopilot.ai.payload`) and never
decides whether a suggestion is trustworthy (see :mod:`dqcopilot.ai.grounding`).

The application must keep working with no API key, so every call returns an
:class:`AiResult` describing what happened rather than raising.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from dqcopilot.config import Settings, get_settings
from dqcopilot.logging_conf import get_logger

logger = get_logger(__name__)

ResponseT = TypeVar("ResponseT", bound=BaseModel)


class AiUnavailableError(Exception):
    """The AI features cannot run (no key, or the SDK is not installed)."""


@dataclass(slots=True)
class AiUsage:
    """Token usage and latency for one call - never any content."""

    task: str
    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: int = 0
    succeeded: bool = False
    error_type: str | None = None

    def as_log_extra(self) -> dict[str, Any]:
        """Return a log-safe dictionary describing the call."""
        return {
            "ai_task": self.task,
            "ai_model": self.model,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "latency_ms": self.latency_ms,
            "succeeded": self.succeeded,
            "error_type": self.error_type,
        }


@dataclass(slots=True)
class AiResult[T: BaseModel]:
    """The outcome of one AI call."""

    usage: AiUsage
    data: T | None = None
    error: str = ""

    @property
    def ok(self) -> bool:
        """True when a validated response is available."""
        return self.data is not None


@dataclass(slots=True)
class AiTelemetry:
    """Running totals for the calls made in one session."""

    calls: list[AiUsage] = field(default_factory=list)

    def record(self, usage: AiUsage) -> None:
        """Add one call to the totals."""
        self.calls.append(usage)

    @property
    def total_input_tokens(self) -> int:
        """Total input tokens across every recorded call."""
        return sum(call.input_tokens for call in self.calls)

    @property
    def total_output_tokens(self) -> int:
        """Total output tokens across every recorded call."""
        return sum(call.output_tokens for call in self.calls)

    @property
    def total_latency_ms(self) -> int:
        """Total latency across every recorded call."""
        return sum(call.latency_ms for call in self.calls)

    @property
    def failures(self) -> int:
        """Number of calls that did not return a validated response."""
        return sum(1 for call in self.calls if not call.succeeded)

    def summary(self) -> dict[str, int]:
        """Return a compact summary for the dashboard."""
        return {
            "calls": len(self.calls),
            "input_tokens": self.total_input_tokens,
            "output_tokens": self.total_output_tokens,
            "latency_ms": self.total_latency_ms,
            "failures": self.failures,
        }


class AnthropicClient:
    """Thin, defensive wrapper around ``client.messages.parse``."""

    def __init__(self, settings: Settings | None = None, client: Any | None = None) -> None:
        """Create a client.

        Args:
            settings: Optional settings override.
            client: An already-built Anthropic SDK client, used by the tests to inject
                a stub. When omitted, one is created from the configured API key.
        """
        self.settings = settings or get_settings()
        self.telemetry = AiTelemetry()
        self._client = client

    @property
    def enabled(self) -> bool:
        """True when the client can actually make calls."""
        return self._client is not None or self.settings.ai_enabled

    def _ensure_client(self) -> Any:
        if self._client is not None:
            return self._client

        if not self.settings.ai_enabled:
            raise AiUnavailableError(
                "No ANTHROPIC_API_KEY is configured, so AI suggestions are unavailable. "
                "Every deterministic check still runs."
            )
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover - the dependency is declared
            raise AiUnavailableError("The anthropic package is not installed.") from exc

        key = self.settings.anthropic_api_key
        assert key is not None  # guaranteed by settings.ai_enabled
        self._client = anthropic.Anthropic(
            api_key=key.get_secret_value(),
            timeout=self.settings.ai_timeout_seconds,
            max_retries=1,
        )
        return self._client

    def parse(
        self,
        task: str,
        system: str,
        user_content: str,
        output_format: type[ResponseT],
    ) -> AiResult[ResponseT]:
        """Make one structured call and validate the response.

        Args:
            task: Short identifier used for logging and telemetry.
            system: The system prompt.
            user_content: The user message (already reduced to metadata).
            output_format: The Pydantic model the response must conform to.

        Returns:
            An :class:`AiResult`. Failures are reported, never raised.
        """
        usage = AiUsage(task=task, model=self.settings.anthropic_model)
        started = time.perf_counter()

        try:
            client = self._ensure_client()
            response = client.messages.parse(
                model=self.settings.anthropic_model,
                max_tokens=self.settings.ai_max_output_tokens,
                system=system,
                messages=[{"role": "user", "content": user_content}],
                output_format=output_format,
            )
        except AiUnavailableError as exc:
            return self._fail(usage, started, "unavailable", str(exc))
        except Exception as exc:  # noqa: BLE001 - the UI must survive any SDK failure
            return self._fail(usage, started, type(exc).__name__, self._safe_message(exc))

        usage.latency_ms = int((time.perf_counter() - started) * 1000)
        usage.input_tokens = int(getattr(response.usage, "input_tokens", 0) or 0)
        usage.output_tokens = int(getattr(response.usage, "output_tokens", 0) or 0)

        parsed = getattr(response, "parsed_output", None)
        if parsed is None:
            return self._fail(
                usage, started, "empty_response", "The model returned no structured output."
            )

        # The SDK already validated, but the response is still untrusted input: a
        # second explicit validation keeps the contract obvious and covers stub clients.
        try:
            data = output_format.model_validate(
                parsed if isinstance(parsed, dict) else parsed.model_dump()
            )
        except ValidationError as exc:
            return self._fail(
                usage, started, "ValidationError", f"{len(exc.errors())} schema violation(s)."
            )

        usage.succeeded = True
        self.telemetry.record(usage)
        logger.info("AI call completed", extra=usage.as_log_extra())
        return AiResult(usage=usage, data=data)

    # ------------------------------------------------------------------ internals

    def _fail(self, usage: AiUsage, started: float, error_type: str, message: str) -> AiResult[Any]:
        usage.latency_ms = int((time.perf_counter() - started) * 1000)
        usage.succeeded = False
        usage.error_type = error_type
        self.telemetry.record(usage)
        logger.warning("AI call failed", extra=usage.as_log_extra())
        return AiResult(usage=usage, data=None, error=message)

    @staticmethod
    def _safe_message(exc: Exception) -> str:
        """Return an error message safe to show, without echoing the request."""
        name = type(exc).__name__
        known = {
            "AuthenticationError": "The Anthropic API key was rejected.",
            "PermissionDeniedError": "The Anthropic API key lacks permission for this model.",
            "NotFoundError": "The configured Anthropic model was not found.",
            "RateLimitError": "The Anthropic API rate limit was reached. Try again shortly.",
            "APITimeoutError": "The Anthropic API did not respond in time.",
            "APIConnectionError": "Could not reach the Anthropic API. Check the network.",
        }
        return known.get(name, f"The Anthropic API call failed ({name}).")


def build_user_content(payload: dict[str, Any] | list[Any]) -> str:
    """Serialise a payload as compact, deterministic JSON for the prompt."""
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
