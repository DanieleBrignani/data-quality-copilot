"""The review session: proposals, human decisions, and the cleaned result.

This object is the workflow the UI drives. It holds the analysis, the proposals derived
from it and the user's decisions, and it can produce the cleaned dataset on demand.
Keeping it free of Streamlit means the whole approval flow is testable directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime

import pandas as pd

from dqcopilot.corrections.applier import apply_corrections, approved_proposals
from dqcopilot.corrections.proposer import propose_corrections
from dqcopilot.logging_conf import get_logger
from dqcopilot.models.corrections import AppliedChange, CorrectionDecision, CorrectionProposal
from dqcopilot.models.enums import DecisionStatus, FindingSource
from dqcopilot.profiling.profiler import profile_dataset
from dqcopilot.scoring import QualityScore, compute_quality_score
from dqcopilot.services.analysis import AnalysisResult
from dqcopilot.validation.base import CheckContext
from dqcopilot.validation.registry import run_checks

logger = get_logger(__name__)


@dataclass(slots=True)
class CleanedDataset:
    """The outcome of applying every approved correction."""

    frame: pd.DataFrame
    applied: list[AppliedChange]
    score_before: QualityScore
    score_after: QualityScore
    rows_before: int
    rows_after: int
    produced_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    @property
    def score_delta(self) -> float:
        """Change in the overall quality score, positive when the data improved."""
        return round(self.score_after.overall - self.score_before.overall, 2)

    @property
    def rows_removed(self) -> int:
        """Number of rows dropped by de-duplication."""
        return self.rows_before - self.rows_after

    @property
    def total_cells_changed(self) -> int:
        """Total number of cell-level changes applied."""
        return sum(change.rows_changed for change in self.applied)


@dataclass(slots=True)
class ReviewSession:
    """An analysis plus the corrections offered for it and the decisions taken."""

    analysis: AnalysisResult
    proposals: list[CorrectionProposal] = field(default_factory=list)
    decisions: dict[str, CorrectionDecision] = field(default_factory=dict)
    reviewer: str = "demo-user"

    @classmethod
    def from_analysis(cls, analysis: AnalysisResult, reviewer: str = "demo-user") -> ReviewSession:
        """Build a session, deriving proposals from the analysis findings."""
        proposals = propose_corrections(
            analysis.findings.sorted(), analysis.frame, analysis.profile
        )
        logger.info(
            "Review session opened",
            extra={
                "analysis_id": analysis.analysis_id,
                "findings": len(analysis.findings),
                "proposals": len(proposals),
            },
        )
        return cls(analysis=analysis, proposals=proposals, reviewer=reviewer)

    # ------------------------------------------------------------------ decisions

    def proposal(self, proposal_id: str) -> CorrectionProposal | None:
        """Return a proposal by id."""
        for proposal in self.proposals:
            if proposal.proposal_id == proposal_id:
                return proposal
        return None

    def approve(self, proposal_id: str) -> None:
        """Approve one proposal.

        Raises:
            KeyError: If the proposal does not belong to this session.
        """
        self._require(proposal_id)
        self.decisions[proposal_id] = CorrectionDecision.approve(proposal_id, self.reviewer)

    def reject(self, proposal_id: str) -> None:
        """Reject one proposal.

        Raises:
            KeyError: If the proposal does not belong to this session.
        """
        self._require(proposal_id)
        self.decisions[proposal_id] = CorrectionDecision.reject(proposal_id, self.reviewer)

    def set_decision(self, proposal_id: str, approved: bool) -> None:
        """Approve or reject depending on ``approved``."""
        self.approve(proposal_id) if approved else self.reject(proposal_id)

    def status(self, proposal_id: str) -> DecisionStatus:
        """Return the current decision status of a proposal."""
        decision = self.decisions.get(proposal_id)
        return decision.status if decision else DecisionStatus.PENDING

    def _require(self, proposal_id: str) -> CorrectionProposal:
        proposal = self.proposal(proposal_id)
        if proposal is None:
            raise KeyError(f"Unknown proposal id: {proposal_id}")
        return proposal

    # ------------------------------------------------------------------ views

    @property
    def actionable(self) -> list[CorrectionProposal]:
        """Proposals that can actually be applied (excludes manual-review items)."""
        return [proposal for proposal in self.proposals if proposal.action.is_applicable]

    @property
    def manual_only(self) -> list[CorrectionProposal]:
        """Proposals that only describe a decision for a human."""
        return [proposal for proposal in self.proposals if not proposal.action.is_applicable]

    @property
    def approved(self) -> list[CorrectionProposal]:
        """Proposals the user approved."""
        return [
            proposal
            for proposal in self.proposals
            if self.status(proposal.proposal_id) is DecisionStatus.APPROVED
        ]

    @property
    def rejected(self) -> list[CorrectionProposal]:
        """Proposals the user rejected."""
        return [
            proposal
            for proposal in self.proposals
            if self.status(proposal.proposal_id) is DecisionStatus.REJECTED
        ]

    @property
    def pending(self) -> list[CorrectionProposal]:
        """Proposals with no decision yet."""
        return [
            proposal
            for proposal in self.proposals
            if self.status(proposal.proposal_id) is DecisionStatus.PENDING
        ]

    def decision_counts(self) -> dict[str, int]:
        """Return a summary of decisions taken, for the dashboard."""
        return {
            "total": len(self.proposals),
            "actionable": len(self.actionable),
            "approved": len(self.approved),
            "rejected": len(self.rejected),
            "pending": len(self.pending),
            "manual_review": len(self.manual_only),
            "ai_suggested": len([p for p in self.proposals if p.source is FindingSource.AI]),
        }

    # ------------------------------------------------------------------ apply

    def has_approved_work(self) -> bool:
        """True when at least one applicable correction is approved."""
        return bool(approved_proposals(self.proposals, self.decisions))

    def build_cleaned_dataset(self) -> CleanedDataset:
        """Apply every approved correction and re-score the result.

        The dataset is re-profiled and re-checked afterwards so the reported improvement
        is measured, not asserted.
        """
        original = self.analysis.frame
        cleaned, applied = apply_corrections(original, self.proposals, self.decisions)

        profile = profile_dataset(cleaned, source_name=self.analysis.source_name)
        findings = run_checks(
            CheckContext(
                frame=cleaned,
                profile=profile,
                rules=self.analysis.rules,
                notes=list(self.analysis.notes),
            )
        )
        score_after = compute_quality_score(profile, findings)

        result = CleanedDataset(
            frame=cleaned,
            applied=applied,
            score_before=self.analysis.score,
            score_after=score_after,
            rows_before=int(original.shape[0]),
            rows_after=int(cleaned.shape[0]),
        )
        logger.info(
            "Cleaned dataset produced",
            extra={
                "analysis_id": self.analysis.analysis_id,
                "changes_applied": len(applied),
                "rows_removed": result.rows_removed,
                "score_before": result.score_before.overall,
                "score_after": result.score_after.overall,
            },
        )
        return result
