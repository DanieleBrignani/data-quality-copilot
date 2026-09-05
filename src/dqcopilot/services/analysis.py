"""Analysis orchestration: the single entry point the UI and the tests both call.

Keeping the flow here (rather than in the Streamlit page) is what makes the
end-to-end path testable without a browser.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime

import pandas as pd

from dqcopilot.config import Settings, get_settings
from dqcopilot.ingestion.loader import LoadedDataset, load_tabular
from dqcopilot.logging_conf import get_logger
from dqcopilot.models.findings import FindingSet
from dqcopilot.models.profile import DatasetProfile
from dqcopilot.profiling.profiler import profile_dataset
from dqcopilot.rules.loader import load_rules_or_none
from dqcopilot.rules.models import BusinessRuleSet
from dqcopilot.scoring import QualityScore, compute_quality_score
from dqcopilot.validation.base import CheckContext
from dqcopilot.validation.registry import run_checks

logger = get_logger(__name__)


@dataclass(slots=True)
class AnalysisResult:
    """Everything produced by one analysis run."""

    analysis_id: str
    source_name: str
    frame: pd.DataFrame
    profile: DatasetProfile
    findings: FindingSet
    score: QualityScore
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    duration_seconds: float = 0.0
    notes: list[str] = field(default_factory=list)
    row_count: int = 0
    column_count: int = 0
    size_bytes: int = 0
    #: Kept so the cleaned dataset is re-checked against exactly the same rules.
    rules: BusinessRuleSet | None = None

    def summary(self) -> dict[str, object]:
        """Return a compact, log-safe summary (no cell values)."""
        return {
            "analysis_id": self.analysis_id,
            "rows": self.row_count,
            "columns": self.column_count,
            "findings": len(self.findings),
            "quality_score": self.score.overall,
            "duration_seconds": round(self.duration_seconds, 3),
        }


def analyze_bytes(
    content: bytes,
    filename: str,
    settings: Settings | None = None,
) -> AnalysisResult:
    """Load, profile and validate an uploaded file.

    Args:
        content: Raw bytes of the uploaded file.
        filename: Original filename.
        settings: Optional settings override.

    Returns:
        An :class:`AnalysisResult`.

    Raises:
        IngestionError: Propagated from the ingestion layer for invalid input.
    """
    settings = settings or get_settings()
    dataset = load_tabular(content, filename, settings=settings)
    rules = load_rules_or_none(settings.rules_file)
    return analyze_dataset(dataset, rules=rules)


def analyze_dataset(
    dataset: LoadedDataset,
    rules: BusinessRuleSet | None = None,
) -> AnalysisResult:
    """Profile and validate an already parsed dataset.

    Args:
        dataset: The parsed dataset.
        rules: Optional business rule set. When ``None`` the business rule check is a
            no-op and every other check still runs.
    """
    started = time.perf_counter()
    analysis_id = uuid.uuid4().hex

    profile = profile_dataset(dataset.frame, source_name=dataset.source_name)
    context = CheckContext(
        frame=dataset.frame,
        profile=profile,
        rules=rules,
        notes=list(dataset.notes),
    )
    findings = run_checks(context)
    score = compute_quality_score(profile, findings)

    result = AnalysisResult(
        analysis_id=analysis_id,
        source_name=dataset.source_name,
        frame=dataset.frame,
        profile=profile,
        findings=findings,
        score=score,
        duration_seconds=time.perf_counter() - started,
        notes=list(dataset.notes),
        row_count=dataset.row_count,
        column_count=dataset.column_count,
        size_bytes=dataset.size_bytes,
        rules=rules,
    )

    logger.info("Analysis completed", extra=result.summary())
    return result
