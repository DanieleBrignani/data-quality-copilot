"""Build the downloadable data quality report.

The report is a single self-contained HTML file: no external CSS, no fonts, no scripts.
That matters for a deliverable someone will email around or attach to a ticket - it
renders the same in five years as it does today, and it leaks nothing by fetching
remote assets.

Unlike the database, the report *does* contain example values, because it is generated
in the user's session and downloaded by them. The split is deliberate and documented:
the stored audit trail is metadata, the report the user holds is complete.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, select_autoescape

from dqcopilot import __version__
from dqcopilot.models.enums import DecisionStatus, FindingSource, Severity
from dqcopilot.services.review import CleanedDataset, ReviewSession

TEMPLATE_DIR = Path(__file__).parent / "templates"


def _environment() -> Environment:
    """Build the Jinja environment with autoescaping on."""
    return Environment(
        loader=FileSystemLoader(TEMPLATE_DIR),
        autoescape=select_autoescape(["html"]),
        trim_blocks=True,
        lstrip_blocks=True,
    )


def build_report_context(
    review: ReviewSession,
    cleaned: CleanedDataset | None = None,
    ai_enabled: bool = False,
) -> dict[str, Any]:
    """Assemble everything the report template needs.

    Kept separate from rendering so the contents can be asserted in tests without
    parsing HTML.
    """
    analysis = review.analysis
    score = analysis.score

    deterministic = [
        finding
        for finding in analysis.findings.sorted()
        if finding.source is FindingSource.DETERMINISTIC
    ]
    ai_findings = [
        finding for finding in analysis.findings.sorted() if finding.source is FindingSource.AI
    ]

    return {
        "analysis_id": analysis.analysis_id,
        "source_name": analysis.source_name,
        "row_count": f"{analysis.row_count:,}",
        "column_count": f"{analysis.column_count:,}",
        "generated_at": datetime.now(UTC).strftime("%Y-%m-%d %H:%M"),
        "app_version": __version__,
        "ai_enabled": ai_enabled,
        "notes": analysis.notes,
        "score": {
            "overall": f"{score.overall:.1f}",
            "grade": score.grade,
            "dataset_penalty": f"{score.dataset_penalty:.1f}",
        },
        "severity_counts": [
            (severity.value, score.severity_counts.get(severity.value, 0))
            for severity in Severity
            if score.severity_counts.get(severity.value, 0)
        ],
        "columns": [
            {
                "name": column.name,
                "semantic_type": column.semantic_type.value,
                "missing_count": f"{column.missing_count:,}",
                "missing_pct": f"{column.missing_ratio:.1%}",
                "unique_count": f"{column.unique_count:,}",
                "score": round(score.column_score(column.name), 1),
            }
            for column in analysis.profile.columns
        ],
        "deterministic_findings": [
            {
                "severity": finding.severity.value,
                "title": finding.title,
                "column": finding.column or "(whole dataset)",
                "affected_rows": f"{finding.affected_rows:,}",
                "explanation": finding.explanation,
            }
            for finding in deterministic
        ],
        "ai_findings": [
            {
                "column": finding.column or "(whole dataset)",
                "title": finding.title,
                "explanation": finding.explanation,
                "confidence": f"{finding.confidence:.0%}",
            }
            for finding in ai_findings
        ],
        "decisions": review.decision_counts(),
        "proposals": [
            {
                "status": review.status(proposal.proposal_id).value,
                "action": proposal.action.value,
                "column": proposal.column or "(whole dataset)",
                "affected_rows": f"{proposal.affected_rows:,}",
                "description": proposal.description,
                "destructive": proposal.destructive,
                "ai": proposal.source is FindingSource.AI,
            }
            for proposal in review.proposals
        ],
        "applied": [
            {
                "action": change.action.value,
                "column": change.column or "(whole dataset)",
                "rows_changed": f"{change.rows_changed:,}",
                "applied_at": change.applied_at.strftime("%Y-%m-%d %H:%M:%S"),
            }
            for change in (cleaned.applied if cleaned else [])
        ],
        "cleaned": {
            "score_before": f"{cleaned.score_before.overall:.1f}" if cleaned else "-",
            "score_after": f"{cleaned.score_after.overall:.1f}" if cleaned else "-",
            "score_delta": f"{cleaned.score_delta:+.1f}" if cleaned else "-",
            "rows_before": f"{cleaned.rows_before:,}" if cleaned else "-",
            "rows_after": f"{cleaned.rows_after:,}" if cleaned else "-",
        },
    }


def render_html_report(
    review: ReviewSession,
    cleaned: CleanedDataset | None = None,
    ai_enabled: bool = False,
) -> str:
    """Render the full HTML report as a string."""
    context = build_report_context(review, cleaned, ai_enabled)
    return _environment().get_template("report.html").render(**context)


def report_bytes(
    review: ReviewSession,
    cleaned: CleanedDataset | None = None,
    ai_enabled: bool = False,
) -> bytes:
    """Render the HTML report as UTF-8 bytes, ready to download."""
    return render_html_report(review, cleaned, ai_enabled).encode("utf-8")


def _decision_metadata(review: ReviewSession, proposal_id: str) -> dict[str, str | None]:
    """Return who decided a proposal and when, or nulls when it is still pending."""
    decision = review.decisions.get(proposal_id)
    if decision is None:
        return {"decided_at": None, "decided_by": None}
    return {
        "decided_at": decision.decided_at.isoformat() if decision.decided_at else None,
        "decided_by": decision.decided_by,
    }


def audit_log_json(
    review: ReviewSession,
    cleaned: CleanedDataset | None = None,
) -> bytes:
    """Build the machine-readable audit log of every decision and change.

    This is the artefact that answers "who approved what, and what did it do" without
    needing database access.
    """
    analysis = review.analysis
    payload = {
        "analysis_id": analysis.analysis_id,
        "source_name": analysis.source_name,
        "analysed_at": analysis.created_at.isoformat(),
        "exported_at": datetime.now(UTC).isoformat(),
        "app_version": __version__,
        "reviewer": review.reviewer,
        "dataset": {
            "rows": analysis.row_count,
            "columns": analysis.column_count,
            "quality_score": analysis.score.overall,
        },
        "decisions": [
            {
                "proposal_id": proposal.proposal_id,
                "finding_id": proposal.finding_id,
                "action": proposal.action.value,
                "column": proposal.column,
                "source": proposal.source.value,
                "destructive": proposal.destructive,
                "affected_rows": proposal.affected_rows,
                "status": review.status(proposal.proposal_id).value,
                **_decision_metadata(review, proposal.proposal_id),
            }
            for proposal in review.proposals
        ],
        "applied_changes": [
            {
                "proposal_id": change.proposal_id,
                "action": change.action.value,
                "column": change.column,
                "rows_changed": change.rows_changed,
                "applied_at": change.applied_at.isoformat(),
            }
            for change in (cleaned.applied if cleaned else [])
        ],
        "outcome": {
            "score_before": cleaned.score_before.overall if cleaned else None,
            "score_after": cleaned.score_after.overall if cleaned else None,
            "rows_before": cleaned.rows_before if cleaned else None,
            "rows_after": cleaned.rows_after if cleaned else None,
            "cells_changed": cleaned.total_cells_changed if cleaned else 0,
        },
        "disclaimer": (
            "The quality score is a transparent heuristic defined by this project, not a "
            "validated data quality metric."
        ),
    }
    return json.dumps(payload, indent=2, ensure_ascii=False).encode("utf-8")


def pending_decision_count(review: ReviewSession) -> int:
    """Number of proposals still awaiting a decision."""
    return sum(
        1
        for proposal in review.actionable
        if review.status(proposal.proposal_id) is DecisionStatus.PENDING
    )
