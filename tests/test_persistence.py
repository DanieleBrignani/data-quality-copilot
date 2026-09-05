from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from sqlalchemy.orm import Session, sessionmaker

from dqcopilot.config import Settings
from dqcopilot.db import (
    Analysis,
    AuditEvent,
    approved_changes,
    audit_trail,
    build_engine,
    check_connection,
    content_hash,
    create_all,
    recent_analyses,
    record_ai_call,
    redact_details,
    require_analysis,
    save_analysis,
    save_cleaned_dataset,
    save_review,
)
from dqcopilot.db.session import DatabaseUnavailableError, session_scope
from dqcopilot.models import CorrectionAction, DecisionStatus
from dqcopilot.services import ReviewSession, analyze_bytes
from dqcopilot.services.persistence import (
    store_analysis,
    store_cleaned_dataset,
    store_download,
    store_review,
)


@pytest.fixture
def db_settings(tmp_path: Path) -> Settings:
    """Settings pointing at a throwaway SQLite file."""
    return Settings(
        anthropic_api_key=None,
        database_url=f"sqlite+pysqlite:///{(tmp_path / 'test.db').as_posix()}",
        persistence_enabled=True,
    )


@pytest.fixture
def factory(db_settings: Settings) -> sessionmaker[Session]:
    engine = build_engine(db_settings)
    create_all(engine)
    return sessionmaker(bind=engine, expire_on_commit=False, future=True)


@pytest.fixture
def messy_csv() -> bytes:
    frame = pd.DataFrame(
        {
            "customer_id": ["C1", "C2", "C3", "C2"],
            "full_name": ["Alice", " Bob ", "bob", " Bob "],
            "email": ["a@example.com", "broken@", "c@example.com", "broken@"],
            "country": ["FR", "fr", "DE", "fr"],
        }
    )
    return frame.to_csv(index=False).encode()


def analyse(content: bytes, settings: Settings) -> ReviewSession:
    result = analyze_bytes(content, "customers.csv", settings=settings)
    return ReviewSession.from_analysis(result)


class TestRedaction:
    def test_example_values_are_replaced_by_counts(self) -> None:
        details = {"examples": ["alice@example.com", "bob@example.com"], "missing_ratio": 0.25}
        redacted = redact_details(details)
        assert "examples" not in redacted
        assert redacted["examples_count"] == 2
        assert redacted["missing_ratio"] == 0.25

    def test_variant_groups_are_redacted(self) -> None:
        details = {"variant_groups": {"fr": ["FR", "fr"]}}
        assert redact_details(details) == {"variant_groups_count": 1}

    def test_opting_in_keeps_the_values(self) -> None:
        details = {"examples": ["alice@example.com"]}
        assert redact_details(details, persist_examples=True) == details

    def test_non_sensitive_keys_pass_through(self) -> None:
        details = {"duplicate_groups": 3, "comparison": "trimmed"}
        assert redact_details(details) == details


class TestContentHash:
    def test_is_stable_and_distinct(self) -> None:
        assert content_hash(b"abc") == content_hash(b"abc")
        assert content_hash(b"abc") != content_hash(b"abd")
        assert len(content_hash(b"abc")) == 64


class TestSaveAnalysis:
    def test_stores_analysis_columns_and_findings(
        self, factory: sessionmaker[Session], messy_csv: bytes, db_settings: Settings
    ) -> None:
        result = analyze_bytes(messy_csv, "customers.csv", settings=db_settings)

        with session_scope(factory) as session:
            save_analysis(session, result, content=messy_csv, settings=db_settings)

        with session_scope(factory) as session:
            stored = require_analysis(session, result.analysis_id)
            assert stored.row_count == 4
            assert stored.column_count == 4
            assert len(stored.columns) == 4
            assert len(stored.findings) == len(result.findings)
            assert stored.quality_score == result.score.overall
            assert stored.content_hash == content_hash(messy_csv)
            assert stored.app_version

    def test_does_not_store_raw_cell_values(
        self, factory: sessionmaker[Session], messy_csv: bytes, db_settings: Settings
    ) -> None:
        result = analyze_bytes(messy_csv, "customers.csv", settings=db_settings)
        with session_scope(factory) as session:
            save_analysis(session, result, content=messy_csv, settings=db_settings)

        with session_scope(factory) as session:
            stored = require_analysis(session, result.analysis_id)
            serialised = "".join(str(finding.details) for finding in stored.findings)
            assert "broken@" not in serialised
            assert "Alice" not in serialised

    def test_writes_an_audit_event(
        self, factory: sessionmaker[Session], messy_csv: bytes, db_settings: Settings
    ) -> None:
        result = analyze_bytes(messy_csv, "customers.csv", settings=db_settings)
        with session_scope(factory) as session:
            save_analysis(session, result, content=messy_csv, settings=db_settings)

        with session_scope(factory) as session:
            events = audit_trail(session, result.analysis_id)
            assert [event.event_type for event in events] == ["analysis.completed"]
            assert "Analysed" in events[0].summary

    def test_column_scores_are_stored(
        self, factory: sessionmaker[Session], messy_csv: bytes, db_settings: Settings
    ) -> None:
        result = analyze_bytes(messy_csv, "customers.csv", settings=db_settings)
        with session_scope(factory) as session:
            save_analysis(session, result, content=messy_csv, settings=db_settings)

        with session_scope(factory) as session:
            stored = require_analysis(session, result.analysis_id)
            for column in stored.columns:
                assert 0 <= column.column_score <= 100


class TestSaveReview:
    def test_records_approvals_and_rejections(
        self, factory: sessionmaker[Session], messy_csv: bytes, db_settings: Settings
    ) -> None:
        review = analyse(messy_csv, db_settings)
        actionable = review.actionable
        review.approve(actionable[0].proposal_id)
        review.reject(actionable[1].proposal_id)

        with session_scope(factory) as session:
            save_analysis(session, review.analysis, settings=db_settings)
            save_review(session, review, settings=db_settings)

        with session_scope(factory) as session:
            stored = require_analysis(session, review.analysis.analysis_id)
            statuses = {p.proposal_id: p.decision_status for p in stored.proposals}
            assert statuses[actionable[0].proposal_id] == DecisionStatus.APPROVED.value
            assert statuses[actionable[1].proposal_id] == DecisionStatus.REJECTED.value

    def test_rejections_are_kept_not_discarded(
        self, factory: sessionmaker[Session], messy_csv: bytes, db_settings: Settings
    ) -> None:
        """The trail must be able to show a problem was seen and left alone."""
        review = analyse(messy_csv, db_settings)
        for proposal in review.proposals:
            review.reject(proposal.proposal_id)

        with session_scope(factory) as session:
            save_analysis(session, review.analysis, settings=db_settings)
            save_review(session, review, settings=db_settings)

        with session_scope(factory) as session:
            stored = require_analysis(session, review.analysis.analysis_id)
            assert len(stored.proposals) == len(review.proposals)
            assert all(p.decision_status == "rejected" for p in stored.proposals)

    def test_decision_metadata_is_stored(
        self, factory: sessionmaker[Session], messy_csv: bytes, db_settings: Settings
    ) -> None:
        review = analyse(messy_csv, db_settings)
        target = review.actionable[0]
        review.approve(target.proposal_id)

        with session_scope(factory) as session:
            save_analysis(session, review.analysis, settings=db_settings)
            save_review(session, review, settings=db_settings)

        with session_scope(factory) as session:
            stored = require_analysis(session, review.analysis.analysis_id)
            record = next(p for p in stored.proposals if p.proposal_id == target.proposal_id)
            assert record.decided_by == "demo-user"
            assert record.decided_at is not None

    def test_saving_twice_does_not_duplicate_proposals(
        self, factory: sessionmaker[Session], messy_csv: bytes, db_settings: Settings
    ) -> None:
        review = analyse(messy_csv, db_settings)
        with session_scope(factory) as session:
            save_analysis(session, review.analysis, settings=db_settings)
            save_review(session, review, settings=db_settings)
            save_review(session, review, settings=db_settings)

        with session_scope(factory) as session:
            stored = require_analysis(session, review.analysis.analysis_id)
            assert len(stored.proposals) == len(review.proposals)


class TestSaveCleanedDataset:
    def test_records_every_applied_change(
        self, factory: sessionmaker[Session], messy_csv: bytes, db_settings: Settings
    ) -> None:
        review = analyse(messy_csv, db_settings)
        strip = next(p for p in review.proposals if p.action is CorrectionAction.STRIP_WHITESPACE)
        review.approve(strip.proposal_id)
        cleaned = review.build_cleaned_dataset()

        with session_scope(factory) as session:
            save_analysis(session, review.analysis, settings=db_settings)
            save_review(session, review, settings=db_settings)
            save_cleaned_dataset(session, review, cleaned, settings=db_settings)

        with session_scope(factory) as session:
            changes = approved_changes(session, review.analysis.analysis_id)
            assert len(changes) == 1
            assert changes[0].action == CorrectionAction.STRIP_WHITESPACE.value
            assert changes[0].rows_changed > 0

    def test_audit_trail_tells_the_whole_story(
        self, factory: sessionmaker[Session], messy_csv: bytes, db_settings: Settings
    ) -> None:
        review = analyse(messy_csv, db_settings)
        review.approve(review.actionable[0].proposal_id)
        cleaned = review.build_cleaned_dataset()

        with session_scope(factory) as session:
            save_analysis(session, review.analysis, settings=db_settings)
            save_review(session, review, settings=db_settings)
            save_cleaned_dataset(session, review, cleaned, settings=db_settings)

        with session_scope(factory) as session:
            events = audit_trail(session, review.analysis.analysis_id)
            assert [event.event_type for event in events] == [
                "analysis.completed",
                "review.decided",
                "dataset.cleaned",
            ]


class TestQueries:
    def test_recent_analyses_are_newest_first(
        self, factory: sessionmaker[Session], messy_csv: bytes, db_settings: Settings
    ) -> None:
        ids = []
        for _ in range(3):
            result = analyze_bytes(messy_csv, "customers.csv", settings=db_settings)
            ids.append(result.analysis_id)
            with session_scope(factory) as session:
                save_analysis(session, result, content=messy_csv, settings=db_settings)

        with session_scope(factory) as session:
            recent = recent_analyses(session, limit=2)
            assert len(recent) == 2
            assert recent[0].analysis_id == ids[-1]

    def test_missing_analysis_raises_lookup_error(self, factory: sessionmaker[Session]) -> None:
        with session_scope(factory) as session, pytest.raises(LookupError):
            require_analysis(session, "does-not-exist")

    def test_ai_calls_record_usage_without_content(
        self, factory: sessionmaker[Session], messy_csv: bytes, db_settings: Settings
    ) -> None:
        result = analyze_bytes(messy_csv, "customers.csv", settings=db_settings)
        with session_scope(factory) as session:
            save_analysis(session, result, settings=db_settings)
            record_ai_call(
                session,
                analysis_id=result.analysis_id,
                task="column_meaning",
                model="claude-sonnet-5",
                input_tokens=800,
                output_tokens=250,
                latency_ms=1400,
            )

        with session_scope(factory) as session:
            from dqcopilot.db import ai_usage_totals

            totals = ai_usage_totals(session)
            assert totals == {"calls": 1, "input_tokens": 800, "output_tokens": 250}


class TestGracefulDegradation:
    def test_persistence_can_be_switched_off(self, messy_csv: bytes) -> None:
        settings = Settings(anthropic_api_key=None, persistence_enabled=False)
        result = analyze_bytes(messy_csv, "customers.csv", settings=settings)
        outcome = store_analysis(result, content=messy_csv, settings=settings)
        assert outcome.failed
        assert "disabled" in outcome.detail

    def test_unreachable_database_does_not_raise(self, messy_csv: bytes) -> None:
        """A dead database degrades the audit trail, never the analysis."""
        settings = Settings(
            anthropic_api_key=None,
            database_url="postgresql+psycopg://nobody:nobody@127.0.0.1:1/nothing",
            persistence_enabled=True,
        )
        result = analyze_bytes(messy_csv, "customers.csv", settings=settings)
        engine = build_engine(settings)
        assert check_connection(engine) is False

        outcome = store_analysis(
            result,
            content=messy_csv,
            settings=settings,
            factory=sessionmaker(bind=engine, expire_on_commit=False, future=True),
        )
        assert outcome.failed
        assert "unavailable" in outcome.detail.lower()

    def test_storing_a_review_without_its_analysis_is_reported(
        self, factory: sessionmaker[Session], messy_csv: bytes, db_settings: Settings
    ) -> None:
        review = analyse(messy_csv, db_settings)
        outcome = store_review(review, settings=db_settings, factory=factory)
        assert outcome.failed
        assert "No stored analysis" in outcome.detail


class TestPersistenceService:
    def test_full_happy_path(
        self, factory: sessionmaker[Session], messy_csv: bytes, db_settings: Settings
    ) -> None:
        result = analyze_bytes(messy_csv, "customers.csv", settings=db_settings)
        review = ReviewSession.from_analysis(result)
        review.approve(review.actionable[0].proposal_id)
        cleaned = review.build_cleaned_dataset()

        assert store_analysis(
            result, content=messy_csv, settings=db_settings, factory=factory
        ).stored
        assert store_review(review, settings=db_settings, factory=factory).stored
        assert store_cleaned_dataset(review, cleaned, settings=db_settings, factory=factory).stored
        assert store_download(
            review, "cleaned dataset", settings=db_settings, factory=factory
        ).stored

        with session_scope(factory) as session:
            events = audit_trail(session, result.analysis_id)
            assert [event.event_type for event in events][-1] == "artefact.downloaded"


class TestSessionScope:
    def test_rolls_back_on_error(self, factory: sessionmaker[Session]) -> None:
        with pytest.raises(ValueError), session_scope(factory) as session:
            session.add(Analysis(analysis_id="rollback-me", source_name="x.csv"))
            session.flush()
            raise ValueError("boom")

        with session_scope(factory) as session:
            assert recent_analyses(session) == []

    def test_wraps_database_errors(self, factory: sessionmaker[Session]) -> None:
        with pytest.raises(DatabaseUnavailableError), session_scope(factory) as session:
            # audit_events.event_type is NOT NULL; flushing this must fail.
            session.add(AuditEvent(event_type=None))  # type: ignore[arg-type]
            session.flush()
