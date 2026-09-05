"""The quality dashboard tab."""

from __future__ import annotations

import streamlit as st

from dqcopilot.reporting import (
    column_score_bar,
    datatype_bar,
    decision_bar,
    missing_values_bar,
    score_gauge,
    severity_bar,
    source_split_pie,
)
from dqcopilot.services.review import CleanedDataset, ReviewSession


def render(review: ReviewSession, cleaned: CleanedDataset | None) -> None:
    """Render the dashboard for one review session."""
    analysis = review.analysis
    profile = analysis.profile

    left, right = st.columns([1, 1])
    with left:
        st.plotly_chart(score_gauge(analysis.score), use_container_width=True)
        with st.expander("How is this calculated?"):
            st.markdown(
                "```\n"
                "penalty(finding) = severity_weight x affected_ratio x 100\n"
                "  critical 1.0 | high 0.6 | medium 0.3 | low 0.1 | info 0.0\n\n"
                "column_score = clamp(100 - SUM(column penalties), 0, 100)\n"
                "overall      = clamp(mean(column_scores) - dataset penalties, 0, 100)\n"
                "```\n"
                "Only deterministic findings count. AI suggestions never move the score, "
                "so it stays reproducible without an API key.\n\n"
                "**This is a transparent heuristic defined by this project, not a "
                "validated data quality metric.**"
            )
    with right:
        st.plotly_chart(severity_bar(analysis.score), use_container_width=True)

    _render_headline_metrics(review, cleaned)

    st.plotly_chart(column_score_bar(analysis.score), use_container_width=True)

    left, right = st.columns([1, 1])
    with left:
        st.plotly_chart(missing_values_bar(profile), use_container_width=True)
        st.plotly_chart(datatype_bar(profile), use_container_width=True)
    with right:
        st.plotly_chart(source_split_pie(analysis.findings), use_container_width=True)
        st.plotly_chart(decision_bar(review.decision_counts()), use_container_width=True)

    if cleaned is not None:
        _render_improvement(cleaned)


def _render_headline_metrics(review: ReviewSession, cleaned: CleanedDataset | None) -> None:
    profile = review.analysis.profile
    duplicates = profile.duplicate_row_count
    mixed = sum(
        1
        for column in profile.columns
        if column.semantic_type.value in {"text", "unknown"}
        or column.semantic_type_confidence < 0.7
    )

    columns = st.columns(5)
    columns[0].metric("Missing cells", f"{profile.missing_ratio:.1%}")
    columns[1].metric("Duplicate rows", f"{duplicates:,}")
    columns[2].metric("Ambiguous types", f"{mixed:,}")
    columns[3].metric("Findings", f"{len(review.analysis.findings):,}")
    columns[4].metric(
        "Corrections applied",
        f"{len(cleaned.applied):,}" if cleaned else "0",
    )


def _render_improvement(cleaned: CleanedDataset) -> None:
    st.divider()
    st.subheader("After applying your approved corrections")
    columns = st.columns(4)
    columns[0].metric(
        "Quality score",
        f"{cleaned.score_after.overall:.1f}",
        delta=f"{cleaned.score_delta:+.1f}",
    )
    columns[1].metric("Cells changed", f"{cleaned.total_cells_changed:,}")
    columns[2].metric(
        "Rows",
        f"{cleaned.rows_after:,}",
        delta=f"{-cleaned.rows_removed:,}" if cleaned.rows_removed else "0",
    )
    columns[3].metric("Corrections applied", f"{len(cleaned.applied):,}")

    st.caption(
        "The score after cleaning is measured, not predicted: the cleaned dataset is "
        "re-profiled and re-checked from scratch."
    )
