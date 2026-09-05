"""The downloads tab: cleaned dataset, quality report, audit log."""

from __future__ import annotations

import streamlit as st

from app.ui import state
from dqcopilot.config import Settings
from dqcopilot.export import cleaned_filename, to_csv_bytes, to_xlsx_bytes
from dqcopilot.reporting import audit_log_json, report_bytes
from dqcopilot.services.persistence import store_download
from dqcopilot.services.review import CleanedDataset, ReviewSession


def render(review: ReviewSession, cleaned: CleanedDataset | None, settings: Settings) -> None:
    """Render every downloadable artefact."""
    source = review.analysis.source_name

    st.subheader("Cleaned dataset")
    if cleaned is None:
        st.info(
            "No corrections have been applied yet. Approve what you want on the "
            "**Corrections** tab, then press *Apply*. You can still download the report "
            "and the audit log below.",
            icon="ℹ️",
        )
    else:
        st.caption(
            f"{len(cleaned.applied)} correction(s) applied · "
            f"{cleaned.total_cells_changed:,} cell(s) changed · "
            f"{cleaned.rows_removed:,} row(s) removed · score "
            f"{cleaned.score_before.overall:.1f} → {cleaned.score_after.overall:.1f}."
        )
        left, right = st.columns(2)
        with left:
            st.download_button(
                "Download cleaned CSV",
                data=to_csv_bytes(cleaned.frame),
                file_name=cleaned_filename(source, ".csv"),
                mime="text/csv",
                type="primary",
                on_click=lambda: _record(review, "cleaned CSV", settings),
            )
        with right:
            st.download_button(
                "Download cleaned Excel",
                data=to_xlsx_bytes(cleaned.frame),
                file_name=cleaned_filename(source, ".xlsx"),
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                on_click=lambda: _record(review, "cleaned XLSX", settings),
            )
        st.caption(
            "Exports neutralise cells starting with `=`, `+`, `-` or `@` by prefixing a "
            "quote, so a spreadsheet cannot execute them as formulas."
        )

    st.divider()
    st.subheader("Quality report")
    st.caption(
        "A self-contained HTML file: the profile, every finding, the score and its "
        "formula, and every decision you took. No external assets, so it renders offline."
    )
    st.download_button(
        "Download quality report (HTML)",
        data=report_bytes(review, cleaned, ai_enabled=settings.ai_enabled),
        file_name=f"{_stem(source)}_quality_report.html",
        mime="text/html",
        on_click=lambda: _record(review, "quality report", settings),
    )

    st.divider()
    st.subheader("Audit log")
    st.caption(
        "Machine-readable JSON recording every correction offered, every decision taken "
        "(including rejections) and every change applied."
    )
    st.download_button(
        "Download audit log (JSON)",
        data=audit_log_json(review, cleaned),
        file_name=f"{_stem(source)}_audit_log.json",
        mime="application/json",
        on_click=lambda: _record(review, "audit log", settings),
    )

    notes = state.persistence_notes()
    if notes:
        st.divider()
        st.warning(
            "**The database was unavailable.** Your analysis, corrections and downloads "
            "all worked, but they were not written to the audit database:\n\n"
            + "\n".join(f"- {note}" for note in notes),
            icon="🗄️",
        )


def _stem(source_name: str) -> str:
    return source_name.rsplit(".", 1)[0] or "dataset"


def _record(review: ReviewSession, artefact: str, settings: Settings) -> None:
    outcome = store_download(review, artefact, settings=settings)
    if outcome.failed and outcome.detail and "disabled" not in outcome.detail:
        state.add_persistence_note(outcome.detail)
