"""Plotly figures for the quality dashboard.

Each function returns a figure and never touches Streamlit, so the shapes can be
asserted in tests without a browser.
"""

from __future__ import annotations

import plotly.graph_objects as go

from dqcopilot.models.enums import FindingSource, Severity
from dqcopilot.models.findings import FindingSet
from dqcopilot.models.profile import DatasetProfile
from dqcopilot.scoring import QualityScore

SEVERITY_COLOURS: dict[str, str] = {
    "critical": "#c0392b",
    "high": "#d35400",
    "medium": "#b7950b",
    "low": "#2471a3",
    "info": "#6b7785",
}

_LAYOUT = {
    "margin": {"l": 10, "r": 10, "t": 40, "b": 10},
    "height": 320,
    "paper_bgcolor": "rgba(0,0,0,0)",
    "plot_bgcolor": "rgba(0,0,0,0)",
}


def empty_figure(title: str, message: str) -> go.Figure:
    """Return a placeholder figure stating why there is nothing to plot.

    A bar chart built from empty data is not merely blank: Plotly tries to place the
    ``textposition="outside"`` labels, computes ``-Infinity`` for their position and
    logs an error in the browser console. Beyond the noise, an unexplained empty panel
    reads as a broken chart rather than as good news, so the placeholder says which it is.
    """
    figure = go.Figure()
    figure.add_annotation(
        text=message,
        showarrow=False,
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
        font={"size": 13, "color": "#5b6470"},
    )
    figure.update_layout(
        title=title,
        xaxis={"visible": False},
        yaxis={"visible": False},
        **_LAYOUT,
    )
    return figure


def score_gauge(score: QualityScore) -> go.Figure:
    """A gauge showing the overall quality score."""
    figure = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score.overall,
            number={"suffix": "/100"},
            title={"text": f"Quality score (grade {score.grade})"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#1e8449" if score.overall >= 75 else "#d35400"},
                "steps": [
                    {"range": [0, 40], "color": "#f5d5d0"},
                    {"range": [40, 75], "color": "#fdf0d5"},
                    {"range": [75, 100], "color": "#d9efe0"},
                ],
            },
        )
    )
    figure.update_layout(**_LAYOUT)
    return figure


def severity_bar(score: QualityScore) -> go.Figure:
    """Findings grouped by severity."""
    severities = [severity for severity in Severity if score.severity_counts.get(severity.value)]
    if not severities:
        return empty_figure("Findings by severity", "No deterministic check reported a problem.")

    figure = go.Figure(
        go.Bar(
            x=[severity.value for severity in severities],
            y=[score.severity_counts[severity.value] for severity in severities],
            marker_color=[SEVERITY_COLOURS[severity.value] for severity in severities],
            text=[score.severity_counts[severity.value] for severity in severities],
            textposition="outside",
        )
    )
    figure.update_layout(title="Findings by severity", yaxis_title="findings", **_LAYOUT)
    return figure


def column_score_bar(score: QualityScore, limit: int = 20) -> go.Figure:
    """Per-column quality scores, worst first."""
    ranked = sorted(score.columns, key=lambda entry: entry.score)[:limit]
    figure = go.Figure(
        go.Bar(
            x=[entry.score for entry in ranked],
            y=[entry.column for entry in ranked],
            orientation="h",
            marker_color=["#c0392b" if entry.score < 60 else "#2471a3" for entry in ranked],
            text=[f"{entry.score:.0f}" for entry in ranked],
            textposition="outside",
        )
    )
    figure.update_layout(
        title="Quality score by column (worst first)",
        xaxis={"range": [0, 105], "title": "score"},
        yaxis={"autorange": "reversed"},
        **{**_LAYOUT, "height": max(260, 26 * len(ranked) + 90)},
    )
    return figure


def missing_values_bar(profile: DatasetProfile, limit: int = 20) -> go.Figure:
    """Missing-value percentage per column."""
    ranked = sorted(profile.columns, key=lambda column: -column.missing_ratio)[:limit]
    ranked = [column for column in ranked if column.missing_count > 0]
    if not ranked:
        return empty_figure("Missing values by column", "No column has any missing values.")

    figure = go.Figure(
        go.Bar(
            x=[round(column.missing_ratio * 100, 2) for column in ranked],
            y=[column.name for column in ranked],
            orientation="h",
            marker_color="#b7950b",
            text=[f"{column.missing_ratio:.1%}" for column in ranked],
            textposition="outside",
        )
    )
    figure.update_layout(
        title="Missing values by column",
        xaxis={"title": "% of rows missing", "range": [0, 105]},
        yaxis={"autorange": "reversed"},
        **{**_LAYOUT, "height": max(260, 26 * max(len(ranked), 1) + 90)},
    )
    return figure


def source_split_pie(findings: FindingSet) -> go.Figure:
    """Deterministic findings versus AI suggestions."""
    deterministic = len(findings.by_source(FindingSource.DETERMINISTIC))
    ai_count = len(findings.by_source(FindingSource.AI))
    figure = go.Figure(
        go.Pie(
            labels=["Deterministic checks", "AI suggestions"],
            values=[deterministic, ai_count],
            marker_colors=["#2471a3", "#6b7785"],
            hole=0.45,
            sort=False,
        )
    )
    figure.update_layout(title="Where findings came from", **_LAYOUT)
    return figure


def decision_bar(counts: dict[str, int]) -> go.Figure:
    """Approved, rejected, pending and manual-review corrections."""
    labels = ["approved", "rejected", "pending", "manual_review"]
    colours = ["#1e8449", "#c0392b", "#b7950b", "#6b7785"]
    figure = go.Figure(
        go.Bar(
            x=[label.replace("_", " ") for label in labels],
            y=[counts.get(label, 0) for label in labels],
            marker_color=colours,
            text=[counts.get(label, 0) for label in labels],
            textposition="outside",
        )
    )
    figure.update_layout(title="Correction decisions", yaxis_title="corrections", **_LAYOUT)
    return figure


def datatype_bar(profile: DatasetProfile) -> go.Figure:
    """How many columns carry each inferred type."""
    counts: dict[str, int] = {}
    for column in profile.columns:
        counts[column.semantic_type.value] = counts.get(column.semantic_type.value, 0) + 1
    ordered = sorted(counts.items(), key=lambda item: -item[1])
    figure = go.Figure(
        go.Bar(
            x=[name for name, _ in ordered],
            y=[count for _, count in ordered],
            marker_color="#2471a3",
            text=[count for _, count in ordered],
            textposition="outside",
        )
    )
    figure.update_layout(title="Inferred column types", yaxis_title="columns", **_LAYOUT)
    return figure
