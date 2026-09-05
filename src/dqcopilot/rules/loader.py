"""Load business rules from YAML."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import ValidationError

from dqcopilot.logging_conf import get_logger
from dqcopilot.rules.models import BusinessRuleSet

logger = get_logger(__name__)


class RuleConfigError(Exception):
    """The business rule file is missing, malformed or invalid."""


def load_rules(path: str | Path) -> BusinessRuleSet:
    """Load and validate a rule set from a YAML file.

    Args:
        path: Path to the YAML rule file.

    Returns:
        The validated :class:`BusinessRuleSet`.

    Raises:
        RuleConfigError: If the file cannot be read, parsed or validated.
    """
    rule_path = Path(path)
    if not rule_path.is_file():
        raise RuleConfigError(f"Business rule file not found: {rule_path}")

    try:
        raw = yaml.safe_load(rule_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise RuleConfigError(f"Could not parse {rule_path.name} as YAML: {exc}") from exc

    if raw is None:
        return BusinessRuleSet()
    if not isinstance(raw, dict):
        raise RuleConfigError(f"{rule_path.name} must contain a YAML mapping at the top level.")

    return parse_rules(raw, source=rule_path.name)


def parse_rules(raw: dict[str, object], source: str = "<inline>") -> BusinessRuleSet:
    """Validate an already parsed mapping into a :class:`BusinessRuleSet`.

    Raises:
        RuleConfigError: If validation fails, with a message naming the offending rule.
    """
    try:
        rule_set = BusinessRuleSet.model_validate(raw)
    except ValidationError as exc:
        raise RuleConfigError(_format_validation_error(exc, source)) from exc

    logger.info(
        "Loaded business rules",
        extra={"source": source, "rule_count": len(rule_set.rules)},
    )
    return rule_set


def load_rules_or_none(path: str | Path) -> BusinessRuleSet | None:
    """Load rules, returning ``None`` (and logging) instead of raising.

    Used by the UI, where a broken rule file must not prevent the rest of the analysis.
    """
    try:
        return load_rules(path)
    except RuleConfigError as exc:
        logger.warning("Business rules unavailable", extra={"reason": str(exc)})
        return None


def _format_validation_error(exc: ValidationError, source: str) -> str:
    lines = [f"{source} contains invalid business rules:"]
    for error in exc.errors()[:10]:
        location = " -> ".join(str(part) for part in error["loc"])
        lines.append(f"  - {location}: {error['msg']}")
    return "\n".join(lines)
