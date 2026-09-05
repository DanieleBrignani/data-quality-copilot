"""The correction review and approval panel.

Two rules shape this screen:

* Nothing is pre-approved. Every checkbox starts unticked, so an impatient click on
  "apply" changes nothing.
* Destructive corrections and AI suggestions are visually separated from the safe,
  reversible ones, because those are the two categories where a careless approval
  actually costs something.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app.ui import state
from dqcopilot.models.corrections import CorrectionProposal
from dqcopilot.models.enums import CorrectionAction, DecisionStatus, FindingSource
from dqcopilot.services.review import ReviewSession


def render(review: ReviewSession) -> None:
    """Render the whole corrections tab."""
    actionable = review.actionable
    manual = review.manual_only

    if not actionable and not manual:
        st.success("No corrections are needed for this dataset.", icon="✅")
        return

    _render_summary(review)

    if actionable:
        safe = [p for p in actionable if not p.requires_extra_care]
        risky = [p for p in actionable if p.requires_extra_care]

        st.subheader("Corrections you can apply")
        st.caption(
            "Each one shows exactly what would change. Nothing is applied until you tick "
            "it and press the button at the bottom."
        )

        if safe:
            st.markdown("**Safe and reversible**")
            for proposal in safe:
                _render_proposal(review, proposal)

        if risky:
            st.markdown("**Needs a closer look**")
            st.caption(
                "These either discard information or come from the language model. "
                "Read the preview before approving."
            )
            for proposal in risky:
                _render_proposal(review, proposal)

        _render_apply_button(review)

    if manual:
        st.divider()
        st.subheader(f"Needs a human decision ({len(manual)})")
        st.caption(
            "The application will not guess at these. It explains the problem and leaves "
            "the judgement to you."
        )
        for proposal in manual:
            with st.expander(proposal.title, expanded=False):
                st.write(proposal.description)
                if proposal.rationale:
                    st.caption(proposal.rationale)


def _render_summary(review: ReviewSession) -> None:
    counts = review.decision_counts()
    columns = st.columns(4)
    columns[0].metric("Applicable", counts["actionable"])
    columns[1].metric("Approved", counts["approved"])
    columns[2].metric("Rejected", counts["rejected"])
    columns[3].metric("Manual review", counts["manual_review"])


def _render_proposal(review: ReviewSession, proposal: CorrectionProposal) -> None:
    """Render one proposal with its preview and approve/reject control."""
    status = review.status(proposal.proposal_id)
    icon = {
        DecisionStatus.APPROVED: "✅",
        DecisionStatus.REJECTED: "🚫",
        DecisionStatus.PENDING: "⬜",
    }[status]

    badges = []
    if proposal.destructive:
        badges.append("destructive")
    if proposal.source is FindingSource.AI:
        badges.append("AI suggestion")
    suffix = f" · _{', '.join(badges)}_" if badges else ""

    with st.expander(f"{icon} **{proposal.title}** — {proposal.affected_rows:,} rows{suffix}"):
        st.write(proposal.description)

        if proposal.source is FindingSource.AI:
            st.info(
                "This came from the language model. It was checked against your data "
                "before being offered, but the reasoning is the model's, not a rule.",
                icon="🤖",
            )
        if proposal.destructive:
            st.warning(
                "This discards information. Download the audit log first if you need a "
                "record of the original values.",
                icon="⚠️",
            )

        if proposal.preview:
            st.caption("Preview of what would change (values in [brackets] have whitespace):")
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "row": change.row_index,
                            "column": change.column,
                            "before": change.before,
                            "after": change.after,
                        }
                        for change in proposal.preview
                    ]
                ),
                use_container_width=True,
                hide_index=True,
            )
        elif proposal.action is CorrectionAction.DROP_DUPLICATE_ROWS:
            st.caption(f"{proposal.affected_rows:,} repeated row(s) would be removed.")

        approved = st.checkbox(
            "Approve this correction",
            value=status is DecisionStatus.APPROVED,
            key=f"approve_{proposal.proposal_id}",
        )
        if approved and status is not DecisionStatus.APPROVED:
            review.approve(proposal.proposal_id)
            state.drop_cleaned()
        elif not approved and status is DecisionStatus.APPROVED:
            review.reject(proposal.proposal_id)
            state.drop_cleaned()


def _render_apply_button(review: ReviewSession) -> None:
    st.divider()
    approved = len(review.approved)

    if not review.has_approved_work():
        st.button(
            "Apply approved corrections",
            disabled=True,
            help="Approve at least one correction first.",
        )
        st.caption("Nothing is approved, so the cleaned file would be identical to the input.")
        return

    if st.button(f"Apply {approved} approved correction(s)", type="primary"):
        with st.spinner("Applying corrections and re-checking the dataset…"):
            cleaned = review.build_cleaned_dataset()
            state.set_cleaned(cleaned)
            _persist(review, cleaned)
        st.rerun()


def _persist(review: ReviewSession, cleaned: object) -> None:
    """Write the decisions and applied changes to the audit trail, if possible."""
    from dqcopilot.services.persistence import store_cleaned_dataset

    outcome = store_cleaned_dataset(review, cleaned)  # type: ignore[arg-type]
    if outcome.failed and outcome.detail:
        state.add_persistence_note(outcome.detail)
