"""Reusable Streamlit rendering components."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from dqcopilot.config import Settings
from dqcopilot.models.enums import Severity
from dqcopilot.models.findings import Finding
from dqcopilot.models.profile import DatasetProfile
from dqcopilot.rules import load_rules_or_none
from dqcopilot.scoring import QualityScore

SEVERITY_ICONS: dict[Severity, str] = {
    Severity.CRITICAL: "🔴",
    Severity.HIGH: "🟠",
    Severity.MEDIUM: "🟡",
    Severity.LOW: "🔵",
    Severity.INFO: "⚪",
}


def render_privacy_banner() -> None:
    """Render the mandatory demo warning about uploading real data.

    The wording used to say "only anonymised column metadata" is sent. That was stricter
    than the behaviour: with the default settings a handful of real values leave the
    machine for every column that does not look personal. A promise the code does not
    keep is worse than no promise, so the banner now says what actually happens and the
    AI tab shows the exact payload before anything is sent.
    """
    st.warning(
        "**Demo application - use synthetic data only.** Do not upload confidential, "
        "personal or production data. Uploaded files are held in memory for the duration "
        "of your session and are not stored on disk. Nothing reaches the Anthropic API "
        "unless you press the button on the AI tab - and then it is column names, types "
        "and counts plus a few example values per column, with columns that look personal "
        "or identifying masked first. The tab shows you the exact payload beforehand, and "
        "`AI_SEND_SAMPLES=false` removes the example values entirely.",
        icon="⚠️",
    )


def _render_rule_status(settings: Settings) -> None:
    """Report whether the business rules actually loaded.

    A missing rule file used to fail silently - the checks simply did not run and
    nothing on screen said so. Surfacing it here means a misconfigured deployment is
    visible immediately rather than discovered from a suspiciously clean report.
    """
    st.subheader("Business rules")
    rules = load_rules_or_none(settings.rules_file)

    if rules is None:
        st.warning(
            f"No rule file could be read at `{settings.rules_file}`, so the "
            "configurable business rules are **not** running. Every built-in check "
            "still works. Set `RULES_FILE` to fix this.",
            icon="⚠️",
        )
        return

    enabled = len(rules.enabled_rules())
    st.caption(
        f"`{rules.name}` — **{enabled}** rule(s) active"
        + (f", {len(rules.rules) - enabled} disabled" if len(rules.rules) != enabled else "")
    )


def render_sidebar(settings: Settings) -> None:
    """Render the sidebar with the current configuration and AI status."""
    with st.sidebar:
        st.subheader("Configuration")
        st.caption(
            f"Accepted formats: **CSV, XLSX**  \n"
            f"Maximum upload size: **{settings.max_upload_mb:.0f} MB**  \n"
            f"Maximum rows: **{settings.max_rows:,}**"
        )

        _render_rule_status(settings)

        st.subheader("AI suggestions")
        if settings.ai_enabled:
            st.success(f"Enabled - model `{settings.anthropic_model}`", icon="✅")
        else:
            st.info(
                "Disabled - no `ANTHROPIC_API_KEY` found. Every deterministic check "
                "still runs; only the AI suggestion panel is unavailable.",
                icon="ℹ️",
            )

        st.divider()
        st.caption(
            "Data Quality Copilot is a portfolio demo. The quality score is a "
            "transparent heuristic, not a validated metric."
        )


def render_score_header(score: QualityScore, profile: DatasetProfile, findings: int) -> None:
    """Render the headline metrics for an analysis."""
    columns = st.columns(5)
    columns[0].metric("Quality score", f"{score.overall:.0f}/100", help=score.formula)
    columns[1].metric("Grade", score.grade)
    columns[2].metric("Rows", f"{profile.row_count:,}")
    columns[3].metric("Columns", f"{profile.column_count:,}")
    columns[4].metric("Findings", f"{findings:,}")


def render_severity_counts(score: QualityScore) -> None:
    """Render a compact severity breakdown."""
    counts = score.severity_counts
    parts = [
        f"{SEVERITY_ICONS[severity]} **{counts.get(severity.value, 0)}** {severity.value}"
        for severity in Severity
        if counts.get(severity.value, 0)
    ]
    st.markdown("  ·  ".join(parts) if parts else "No issues detected.")


def render_preview(frame: pd.DataFrame, rows: int) -> None:
    """Render a bounded preview of the dataset."""
    st.caption(f"First {min(rows, len(frame)):,} of {len(frame):,} rows, as read from the file.")
    st.dataframe(frame.head(rows), use_container_width=True, hide_index=False)


def render_profile_table(profile: DatasetProfile, score: QualityScore) -> None:
    """Render the per-column profile as a table."""
    rows = [
        {
            "Column": column.name,
            "Inferred type": column.semantic_type.value,
            "Confidence": f"{column.semantic_type_confidence:.0%}",
            "Missing": f"{column.missing_count:,} ({column.missing_ratio:.1%})",
            "Distinct": f"{column.unique_count:,}",
            "Score": round(score.column_score(column.name), 1),
            "Examples": ", ".join(column.sample_values[:3]),
        }
        for column in profile.columns
    ]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def render_finding(finding: Finding) -> None:
    """Render one finding inside an expander."""
    icon = SEVERITY_ICONS[finding.severity]
    scope = finding.column or "whole dataset"
    label = f"{icon} **{finding.title}** — {finding.severity.value} · {scope}"

    with st.expander(label, expanded=False):
        st.write(finding.explanation)
        meta = st.columns(3)
        meta[0].metric("Affected rows", f"{finding.affected_rows:,}")
        meta[1].metric("Share of rows", f"{finding.affected_ratio:.1%}")
        meta[2].metric("Check", finding.check_id)

        if finding.row_indices:
            preview = ", ".join(str(index) for index in finding.row_indices[:20])
            suffix = " …" if len(finding.row_indices) > 20 else ""
            st.caption(f"Example row numbers: {preview}{suffix}")
        if finding.details:
            with st.popover("Technical details"):
                st.json(finding.details, expanded=True)
