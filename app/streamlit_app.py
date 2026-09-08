"""Data Quality Copilot - Streamlit entry point.

The page is deliberately thin: it collects an upload, hands it to the service layer and
renders the result. All analysis logic lives in :mod:`dqcopilot`, so the whole flow can
be tested without a browser.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import streamlit as st

# Allow `streamlit run app/streamlit_app.py` from a source checkout without installing.
_APP_DIR = Path(__file__).resolve().parent
if str(_APP_DIR.parent) not in sys.path:
    sys.path.insert(0, str(_APP_DIR.parent))

from app.ui import ai_panel, components, corrections, dashboard, downloads, state  # noqa: E402
from dqcopilot.config import get_settings  # noqa: E402
from dqcopilot.db.session import ensure_schema  # noqa: E402
from dqcopilot.ingestion.errors import IngestionError  # noqa: E402
from dqcopilot.logging_conf import configure_logging  # noqa: E402
from dqcopilot.services.analysis import analyze_bytes  # noqa: E402
from dqcopilot.services.persistence import store_analysis  # noqa: E402
from dqcopilot.services.review import ReviewSession  # noqa: E402

st.set_page_config(page_title="Data Quality Copilot", page_icon="🧹", layout="wide")


def main() -> None:
    """Render the application."""
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_json)
    _prepare_database(settings)

    st.title("🧹 Data Quality Copilot")
    st.caption(
        "Upload a CSV or Excel file. The app profiles it, runs deterministic quality "
        "checks, explains what it found and proposes fixes. Nothing changes without "
        "your approval."
    )
    components.render_privacy_banner()
    components.render_sidebar(settings)

    upload = st.file_uploader(
        "Upload a dataset",
        type=["csv", "xlsx"],
        help=f"CSV or XLSX, up to {settings.max_upload_mb:.0f} MB.",
    )

    if upload is None:
        state.clear_all()
        _render_empty_state()
        return

    content = upload.getvalue()
    signature = hashlib.sha256(content).hexdigest()[:16] + upload.name

    if state.upload_signature() != signature:
        _run_analysis(content, upload.name, signature, settings)

    error = state.get_error()
    if error:
        st.error(f"**The file was rejected.** {error}", icon="🚫")
        return

    review = state.get_review()
    if review is None:
        return

    _render_review(review, settings)


@st.cache_resource
def _prepare_database(settings) -> bool:  # noqa: ANN001 - Streamlit page glue
    """Make sure the local SQLite schema exists. Cached so it runs once per process."""
    if not settings.persistence_enabled:
        return False
    return ensure_schema(settings)


def _run_analysis(content: bytes, filename: str, signature: str, settings) -> None:  # noqa: ANN001
    """Analyse an upload and open a review session for it."""
    with st.spinner("Profiling and checking the dataset…"):
        try:
            result = analyze_bytes(content, filename, settings=settings)
        except IngestionError as error:
            state.set_error(str(error))
        else:
            state.set_review(ReviewSession.from_analysis(result))
            outcome = store_analysis(result, content=content, settings=settings)
            if outcome.failed and outcome.detail and "disabled" not in outcome.detail:
                state.add_persistence_note(outcome.detail)
        state.set_upload_signature(signature)


def _render_empty_state() -> None:
    st.info(
        "No file loaded yet. Synthetic demo datasets live in `data/demo/` — they contain "
        "deliberately seeded errors, recorded in `ground_truth.json`, so you can see every "
        "check fire. Regenerate them with `python scripts/generate_demo_data.py`.",
        icon="📄",
    )
    st.info(
        "`data/public/` holds a real open dataset from the Comune di Milano (CC0, no "
        "personal data). Nobody here chose what is wrong with it, which makes it the "
        "honest counterpart to the synthetic files — see `data/public/README.md` for what "
        "it found and what it exposed as missing.",
        icon="🌍",
    )


def _render_review(review: ReviewSession, settings) -> None:  # noqa: ANN001
    """Render the tabs for an analysed dataset."""
    analysis = review.analysis
    cleaned = state.get_cleaned()

    st.divider()
    components.render_score_header(analysis.score, analysis.profile, len(analysis.findings))
    components.render_severity_counts(analysis.score)

    if analysis.notes:
        with st.expander(f"{len(analysis.notes)} note(s) from reading the file"):
            for note in analysis.notes:
                st.write(f"- {note}")

    pending = len([p for p in review.actionable if review.status(p.proposal_id).value == "pending"])
    tabs = st.tabs(
        [
            "📊 Dashboard",
            "👀 Preview",
            "🧬 Column profile",
            f"🔍 Findings ({len(analysis.findings)})",
            f"🛠️ Corrections ({pending} pending)",
            "🤖 AI suggestions",
            "⬇️ Downloads",
        ]
    )

    with tabs[0]:
        dashboard.render(review, cleaned)
    with tabs[1]:
        components.render_preview(analysis.frame, settings.preview_rows)
        if cleaned is not None:
            st.divider()
            st.subheader("Cleaned dataset")
            components.render_preview(cleaned.frame, settings.preview_rows)
    with tabs[2]:
        components.render_profile_table(analysis.profile, analysis.score)
    with tabs[3]:
        _render_findings(review)
    with tabs[4]:
        corrections.render(review)
    with tabs[5]:
        ai_panel.render(review, settings)
    with tabs[6]:
        downloads.render(review, cleaned, settings)


def _render_findings(review: ReviewSession) -> None:
    findings = review.analysis.findings.sorted()
    if not findings:
        st.success("No quality issues were detected by the deterministic checks.", icon="✅")
        return

    deterministic = [f for f in findings if f.source.value == "deterministic"]
    ai_findings = [f for f in findings if f.source.value == "ai"]

    st.caption(
        f"{len(deterministic)} deterministic finding(s) — reproducible, and the only ones "
        "that affect the quality score."
    )
    for finding in deterministic:
        components.render_finding(finding)

    if ai_findings:
        st.divider()
        st.caption(f"{len(ai_findings)} AI suggestion(s) — advisory, and excluded from the score.")
        for finding in ai_findings:
            components.render_finding(finding)


# Streamlit executes this file top to bottom on every interaction, so `main()` is
# called unconditionally rather than behind an `if __name__ == "__main__"` guard.
main()
