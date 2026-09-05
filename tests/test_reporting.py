from __future__ import annotations

import json

import pandas as pd
import pytest

from dqcopilot.config import Settings
from dqcopilot.models import CorrectionAction, FindingSource, Severity
from dqcopilot.models.enums import IssueType
from dqcopilot.models.findings import Finding, make_finding_id
from dqcopilot.reporting import (
    audit_log_json,
    build_report_context,
    column_score_bar,
    datatype_bar,
    decision_bar,
    missing_values_bar,
    pending_decision_count,
    render_html_report,
    report_bytes,
    score_gauge,
    severity_bar,
    source_split_pie,
)
from dqcopilot.services import ReviewSession, analyze_bytes


@pytest.fixture
def settings() -> Settings:
    return Settings(anthropic_api_key=None, persistence_enabled=False)


@pytest.fixture
def messy_csv() -> bytes:
    frame = pd.DataFrame(
        {
            "customer_id": ["C1", "C2", "C3", "C2"],
            "full_name": ["Alice", " Bob ", "bob", " Bob "],
            "email": ["a@example.com", "broken@", "c@example.com", "broken@"],
            "country": ["FR", "fr", "DE", "fr"],
            "revenue": ["1,200.50", "900", "-50", "900"],
        }
    )
    return frame.to_csv(index=False).encode()


@pytest.fixture
def review(messy_csv: bytes, settings: Settings) -> ReviewSession:
    result = analyze_bytes(messy_csv, "customers.csv", settings=settings)
    return ReviewSession.from_analysis(result)


class TestReportContext:
    def test_contains_the_headline_numbers(self, review: ReviewSession) -> None:
        context = build_report_context(review)
        assert context["source_name"] == "customers.csv"
        assert context["row_count"] == "4"
        assert context["column_count"] == "5"
        assert context["analysis_id"] == review.analysis.analysis_id
        assert context["columns"]

    def test_separates_deterministic_from_ai(self, review: ReviewSession) -> None:
        review.analysis.findings.findings.append(
            Finding(
                finding_id=make_finding_id(
                    FindingSource.AI, "ai_x", IssueType.INCONSISTENT_CATEGORY, "country"
                ),
                check_id="ai_x",
                issue_type=IssueType.INCONSISTENT_CATEGORY,
                source=FindingSource.AI,
                severity=Severity.INFO,
                column="country",
                title="AI idea",
                explanation="An advisory suggestion about the country column.",
                row_count=4,
            )
        )
        context = build_report_context(review)
        assert len(context["ai_findings"]) == 1
        assert all(finding["title"] != "AI idea" for finding in context["deterministic_findings"])

    def test_reports_decisions_including_rejections(self, review: ReviewSession) -> None:
        actionable = review.actionable
        review.approve(actionable[0].proposal_id)
        review.reject(actionable[1].proposal_id)

        context = build_report_context(review)
        statuses = {proposal["status"] for proposal in context["proposals"]}
        assert {"approved", "rejected"} <= statuses
        assert context["decisions"]["approved"] == 1
        assert context["decisions"]["rejected"] == 1

    def test_includes_applied_changes_and_score_movement(self, review: ReviewSession) -> None:
        strip = next(p for p in review.proposals if p.action is CorrectionAction.STRIP_WHITESPACE)
        review.approve(strip.proposal_id)
        cleaned = review.build_cleaned_dataset()

        context = build_report_context(review, cleaned)
        assert len(context["applied"]) == 1
        assert context["cleaned"]["score_before"] != "-"
        assert context["cleaned"]["score_delta"].startswith(("+", "-", "0"))

    def test_score_formula_is_carried_into_the_report(self, review: ReviewSession) -> None:
        context = build_report_context(review)
        assert "dataset_penalty" in context["score"]
        assert context["score"]["grade"] in {"A", "B", "C", "D", "E"}


class TestHtmlReport:
    def test_renders_self_contained_html(self, review: ReviewSession) -> None:
        html = render_html_report(review)
        assert html.startswith("<!DOCTYPE html>")
        assert "</html>" in html
        # No external resources: the report must render offline, forever.
        assert "http://" not in html
        assert "https://" not in html
        assert "<script" not in html.lower()

    def test_includes_the_score_formula(self, review: ReviewSession) -> None:
        html = render_html_report(review)
        assert "severity_weight x affected_ratio x 100" in html
        assert "not a validated data quality metric" in html

    def test_includes_the_synthetic_data_disclaimer(self, review: ReviewSession) -> None:
        assert "entirely synthetic" in render_html_report(review)

    def test_escapes_values_from_the_dataset(self, settings: Settings) -> None:
        """A malicious column name must not become markup in the report."""
        frame = pd.DataFrame({"<script>alert(1)</script>": ["a", None]})
        result = analyze_bytes(frame.to_csv(index=False).encode(), "x.csv", settings=settings)
        html = render_html_report(ReviewSession.from_analysis(result))

        assert "<script>alert(1)</script>" not in html
        assert "&lt;script&gt;" in html

    def test_reports_reduced_mode_when_ai_is_off(self, review: ReviewSession) -> None:
        html = render_html_report(review, ai_enabled=False)
        assert "reduced mode" in html

    def test_report_bytes_are_utf8(self, review: ReviewSession) -> None:
        payload = report_bytes(review)
        assert isinstance(payload, bytes)
        assert payload.decode("utf-8").startswith("<!DOCTYPE html>")


class TestAuditLog:
    def test_records_every_decision(self, review: ReviewSession) -> None:
        actionable = review.actionable
        review.approve(actionable[0].proposal_id)
        review.reject(actionable[1].proposal_id)

        payload = json.loads(audit_log_json(review))
        statuses = {entry["status"] for entry in payload["decisions"]}
        assert {"approved", "rejected", "pending"} & statuses
        assert len(payload["decisions"]) == len(review.proposals)

    def test_records_who_decided_and_when(self, review: ReviewSession) -> None:
        target = review.actionable[0]
        review.approve(target.proposal_id)

        payload = json.loads(audit_log_json(review))
        entry = next(e for e in payload["decisions"] if e["proposal_id"] == target.proposal_id)
        assert entry["decided_by"] == "demo-user"
        assert entry["decided_at"] is not None

    def test_records_applied_changes_and_outcome(self, review: ReviewSession) -> None:
        strip = next(p for p in review.proposals if p.action is CorrectionAction.STRIP_WHITESPACE)
        review.approve(strip.proposal_id)
        cleaned = review.build_cleaned_dataset()

        payload = json.loads(audit_log_json(review, cleaned))
        assert len(payload["applied_changes"]) == 1
        assert payload["applied_changes"][0]["action"] == "strip_whitespace"
        assert payload["outcome"]["cells_changed"] > 0
        assert payload["outcome"]["score_after"] is not None

    def test_is_valid_json_and_carries_the_disclaimer(self, review: ReviewSession) -> None:
        payload = json.loads(audit_log_json(review))
        assert payload["app_version"]
        assert "heuristic" in payload["disclaimer"]


class TestPendingCount:
    def test_counts_only_actionable_proposals(self, review: ReviewSession) -> None:
        assert pending_decision_count(review) == len(review.actionable)
        review.approve(review.actionable[0].proposal_id)
        assert pending_decision_count(review) == len(review.actionable) - 1


class TestFigures:
    def test_every_figure_builds(self, review: ReviewSession) -> None:
        analysis = review.analysis
        figures = [
            score_gauge(analysis.score),
            severity_bar(analysis.score),
            column_score_bar(analysis.score),
            missing_values_bar(analysis.profile),
            source_split_pie(analysis.findings),
            decision_bar(review.decision_counts()),
            datatype_bar(analysis.profile),
        ]
        for figure in figures:
            assert figure.data is not None
            assert figure.layout is not None

    def test_score_gauge_shows_the_actual_score(self, review: ReviewSession) -> None:
        figure = score_gauge(review.analysis.score)
        assert figure.data[0].value == review.analysis.score.overall

    def test_column_bar_lists_worst_columns_first(self, review: ReviewSession) -> None:
        figure = column_score_bar(review.analysis.score)
        scores = list(figure.data[0].x)
        assert scores == sorted(scores)

    def test_source_split_counts_both_sources(self, review: ReviewSession) -> None:
        figure = source_split_pie(review.analysis.findings)
        assert sum(figure.data[0].values) == len(review.analysis.findings)
