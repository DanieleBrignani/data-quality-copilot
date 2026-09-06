"""Report generation and dashboard figures."""

from dqcopilot.reporting.figures import (
    column_score_bar,
    datatype_bar,
    decision_bar,
    empty_figure,
    missing_values_bar,
    score_gauge,
    severity_bar,
    source_split_pie,
)
from dqcopilot.reporting.report import (
    audit_log_json,
    build_report_context,
    pending_decision_count,
    render_html_report,
    report_bytes,
)

__all__ = [
    "audit_log_json",
    "build_report_context",
    "column_score_bar",
    "datatype_bar",
    "decision_bar",
    "empty_figure",
    "missing_values_bar",
    "pending_decision_count",
    "render_html_report",
    "report_bytes",
    "score_gauge",
    "severity_bar",
    "source_split_pie",
]
