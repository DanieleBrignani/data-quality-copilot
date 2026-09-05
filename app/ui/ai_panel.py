"""The AI suggestions tab.

Nothing is sent to Anthropic until the user presses the button on this page. That is a
deliberate design choice, not a technical constraint: the privacy promise is much easier
to keep - and to explain - when the network call is an explicit action rather than a
side effect of uploading a file.
"""

from __future__ import annotations

import streamlit as st

from app.ui import components, state
from dqcopilot.ai import AiSuggester, AiSuggestions
from dqcopilot.ai.payload import dataset_payload
from dqcopilot.config import Settings
from dqcopilot.services.review import ReviewSession


def render(review: ReviewSession, settings: Settings) -> None:
    """Render the AI suggestions tab."""
    if not settings.ai_enabled:
        _render_reduced_mode()
        return

    suggestions = state.get_ai()

    st.caption(
        f"Model: `{settings.anthropic_model}` · output capped at "
        f"{settings.ai_max_output_tokens:,} tokens per call."
    )
    _render_what_gets_sent(review, settings)

    if st.button("Generate AI suggestions", type="primary", disabled=suggestions is not None):
        with st.spinner("Asking Anthropic about the ambiguous parts…"):
            result = AiSuggester(settings).run(review.analysis)
            state.set_ai(result)
            _merge_into_review(review, result)
        st.rerun()

    if suggestions is None:
        st.info(
            "No request has been made yet. Deterministic findings are already complete "
            "without this step.",
            icon="ℹ️",
        )
        return

    _render_results(suggestions)


def _render_reduced_mode() -> None:
    st.info(
        "**Reduced mode.** No `ANTHROPIC_API_KEY` is configured, so AI suggestions are "
        "unavailable. Every deterministic check, correction, export and audit record "
        "still works — only this panel is switched off.",
        icon="🔌",
    )
    st.markdown(
        "To enable it, copy `.env.example` to `.env`, set `ANTHROPIC_API_KEY`, and "
        "restart the app. The key is read from the environment and is never written to "
        "disk, logged, or shown in the interface."
    )


def _render_what_gets_sent(review: ReviewSession, settings: Settings) -> None:
    """Show the user exactly what would leave their machine."""
    with st.expander("Show exactly what would be sent to Anthropic"):
        st.markdown(
            "The dataset is **never** sent. Only this shape description is, and columns "
            "that look personal or identifying have their example values replaced by a "
            "format mask (`alice@example.com` becomes `aaaaa@aaaaaaa.aaa`)."
        )
        payload = dataset_payload(review.analysis.profile, review.analysis.frame, settings)
        st.json(payload, expanded=False)
        st.caption(
            f"Sampling: up to {settings.ai_sample_rows} example value(s) per column, "
            f"{settings.ai_max_columns} columns maximum. Set `AI_SEND_SAMPLES=false` to "
            "send counts and types only."
        )


def _merge_into_review(review: ReviewSession, suggestions: AiSuggestions) -> None:
    """Add AI findings and proposals to the session, without touching the score."""
    known = {finding.finding_id for finding in review.analysis.findings.findings}
    for finding in suggestions.findings:
        if finding.finding_id not in known:
            review.analysis.findings.findings.append(finding)

    known_proposals = {proposal.proposal_id for proposal in review.proposals}
    for proposal in suggestions.proposals:
        if proposal.proposal_id not in known_proposals:
            review.proposals.append(proposal)


def _render_results(suggestions: AiSuggestions) -> None:
    if suggestions.errors:
        for error in suggestions.errors:
            st.warning(error, icon="⚠️")

    usage = suggestions.usage
    if usage:
        columns = st.columns(4)
        columns[0].metric("API calls", usage.get("calls", 0))
        columns[1].metric("Input tokens", f"{usage.get('input_tokens', 0):,}")
        columns[2].metric("Output tokens", f"{usage.get('output_tokens', 0):,}")
        columns[3].metric("Total latency", f"{usage.get('latency_ms', 0) / 1000:.1f}s")

    if suggestions.is_empty:
        st.info("The model did not find anything worth suggesting.", icon="🤖")
    else:
        st.success(
            f"{len(suggestions.findings)} suggestion(s), "
            f"{len(suggestions.proposals)} of which can be applied after your approval "
            "on the Corrections tab.",
            icon="🤖",
        )
        for finding in suggestions.findings:
            components.render_finding(finding)

    if suggestions.rejected:
        with st.expander(f"{len(suggestions.rejected)} suggestion(s) were discarded"):
            st.caption(
                "Every model response is checked against your actual data. These did not "
                "survive that check and were never shown as suggestions."
            )
            for reason in suggestions.rejected:
                st.write(f"- {reason}")
