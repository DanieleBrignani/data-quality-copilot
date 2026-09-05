"""Registry and runner for deterministic checks."""

from __future__ import annotations

import time
from collections.abc import Iterable

from dqcopilot.logging_conf import get_logger
from dqcopilot.models.findings import Finding, FindingSet
from dqcopilot.validation.base import Check, CheckContext

logger = get_logger(__name__)

_REGISTRY: dict[str, type[Check]] = {}


def register_check(check_class: type[Check]) -> type[Check]:
    """Class decorator that adds a check to the global registry.

    Raises:
        ValueError: If a check with the same ``check_id`` is already registered.
    """
    check_id = check_class.check_id
    if check_id in _REGISTRY:
        raise ValueError(f"A check with id '{check_id}' is already registered.")
    _REGISTRY[check_id] = check_class
    return check_class


def registered_checks() -> list[type[Check]]:
    """Return all registered check classes, ordered by ``check_id``."""
    return [_REGISTRY[key] for key in sorted(_REGISTRY)]


def run_checks(
    context: CheckContext,
    only: Iterable[str] | None = None,
) -> FindingSet:
    """Run every registered check against ``context``.

    A failing check is logged and skipped rather than aborting the whole analysis:
    one broken rule must never cost the user the other twelve.

    Args:
        context: The dataset under analysis.
        only: Optional iterable of ``check_id`` values to restrict the run.

    Returns:
        A :class:`~dqcopilot.models.findings.FindingSet` with all findings produced.
    """
    wanted = set(only) if only is not None else None
    findings: list[Finding] = []

    for check_class in registered_checks():
        if wanted is not None and check_class.check_id not in wanted:
            continue
        started = time.perf_counter()
        try:
            produced = check_class().run(context)
        except Exception:  # noqa: BLE001 - a broken check must not abort the analysis
            logger.exception("Check failed", extra={"check_id": check_class.check_id})
            continue
        findings.extend(produced)
        logger.debug(
            "Check completed",
            extra={
                "check_id": check_class.check_id,
                "findings": len(produced),
                "duration_seconds": round(time.perf_counter() - started, 4),
            },
        )

    return FindingSet(findings=findings)
