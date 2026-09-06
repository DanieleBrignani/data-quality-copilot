"""Measure the deterministic checks against the seeded ground truth.

This is the test that stops the check suite from quietly rotting: the demo data
generator records every error it injects, and this module asserts that each one is
found by an acceptable check.

A seeded error is matched loosely on purpose. Several checks can legitimately explain
the same defect - a country written ``"France"`` instead of ``"FR"`` is an inconsistent
category *and* a business rule violation - so each seeded issue type maps to the set of
findings that count as having caught it. The mapping is written out explicitly rather
than hidden in a helper, because it is the honest statement of what "detected" means.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import date
from pathlib import Path

import pytest

from dqcopilot.config import Settings
from dqcopilot.demodata import load_ground_truth, write_demo_data
from dqcopilot.models import IssueType
from dqcopilot.rules import load_rules
from dqcopilot.services import analyze_bytes

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RULES_FILE = PROJECT_ROOT / "config" / "business_rules.yaml"

#: seeded issue type -> issue types that count as detecting it.
ACCEPTABLE_DETECTORS: dict[str, set[str]] = {
    IssueType.MISSING_VALUES: {
        IssueType.MISSING_VALUES,
        IssueType.BUSINESS_RULE_VIOLATION,
    },
    IssueType.INVALID_EMAIL: {IssueType.INVALID_EMAIL},
    IssueType.INCONSISTENT_CATEGORY: {
        # A category variant may differ only by case or padding, in which case the
        # more specific formatting checks report it; a genuinely different spelling
        # ("France" for "FR") is caught by the allowed_values business rule.
        IssueType.INCONSISTENT_CATEGORY,
        IssueType.INCONSISTENT_CAPITALIZATION,
        IssueType.LEADING_TRAILING_WHITESPACE,
        IssueType.BUSINESS_RULE_VIOLATION,
        IssueType.MISSING_VALUES,
    },
    IssueType.LEADING_TRAILING_WHITESPACE: {IssueType.LEADING_TRAILING_WHITESPACE},
    IssueType.INCONSISTENT_CAPITALIZATION: {IssueType.INCONSISTENT_CAPITALIZATION},
    IssueType.IMPOSSIBLE_NUMERIC: {
        IssueType.IMPOSSIBLE_NUMERIC,
        IssueType.BUSINESS_RULE_VIOLATION,
    },
    IssueType.NUMERIC_STORED_AS_TEXT: {IssueType.NUMERIC_STORED_AS_TEXT},
    IssueType.MIXED_DATATYPES: {
        # A handful of words in an otherwise numeric column stays below the
        # mixed-type threshold; the numeric check reports it as unparseable values.
        IssueType.MIXED_DATATYPES,
        IssueType.NUMERIC_STORED_AS_TEXT,
    },
    IssueType.INCONSISTENT_DATE_FORMAT: {IssueType.INCONSISTENT_DATE_FORMAT},
    IssueType.INVALID_DATE: {IssueType.INVALID_DATE, IssueType.INCONSISTENT_DATE_FORMAT},
    IssueType.FUTURE_DATE: {IssueType.FUTURE_DATE, IssueType.BUSINESS_RULE_VIOLATION},
    IssueType.EXACT_DUPLICATE_ROWS: {IssueType.EXACT_DUPLICATE_ROWS},
    IssueType.PROBABLE_DUPLICATE_ROWS: {
        IssueType.PROBABLE_DUPLICATE_ROWS,
        IssueType.EXACT_DUPLICATE_ROWS,
    },
    IssueType.BUSINESS_RULE_VIOLATION: {
        IssueType.BUSINESS_RULE_VIOLATION,
        IssueType.IMPOSSIBLE_NUMERIC,
        IssueType.INVALID_DATE,
        IssueType.INCONSISTENT_CAPITALIZATION,
        IssueType.INCONSISTENT_CATEGORY,
    },
    IssueType.CONSTANT_COLUMN: {IssueType.CONSTANT_COLUMN},
}


@pytest.fixture(scope="module")
def demo_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Generate the demo datasets once for the whole module."""
    destination = tmp_path_factory.mktemp("demo")
    write_demo_data(destination)
    return destination


@pytest.fixture(scope="module")
def analyses(demo_dir: Path) -> dict[str, object]:
    """Analyse every demo CSV with the project's real business rules."""
    settings = Settings(anthropic_api_key=None, rules_file=RULES_FILE)
    rules = load_rules(RULES_FILE)
    assert rules.rules, "the demo rule file should not be empty"

    results = {}
    for name in ("customers", "sales", "suppliers"):
        path = demo_dir / f"{name}.csv"
        results[name] = analyze_bytes(path.read_bytes(), path.name, settings=settings)
    return results


class TestGeneratorReproducibility:
    def test_generation_is_deterministic(self, tmp_path: Path) -> None:
        """Every generated file must be byte-identical, the workbook included.

        The XLSX needed three fixes to get here: openpyxl stamps created/modified into
        docProps/core.xml, rewrites modified again inside save(), and the ZIP container
        records the wall-clock time of every entry. Before those, two runs in the same
        second matched and two runs a second apart did not - an intermittent build
        failure rather than an honest one.
        """
        # The reference date is pinned: the demo data deliberately contains dates in
        # the future so the future-date check keeps firing however old the repository
        # gets, which makes the raw output depend on the day it runs. The property
        # worth asserting is determinism given a seed *and* a reference date.
        reference = date(2026, 1, 1)
        first, second = tmp_path / "a", tmp_path / "b"
        write_demo_data(first, reference_date=reference)
        write_demo_data(second, reference_date=reference)
        for name in ("customers.csv", "sales.csv", "suppliers.csv", "ground_truth.json"):
            assert (first / name).read_bytes() == (second / name).read_bytes(), name

        assert (first / "suppliers.xlsx").read_bytes() == (second / "suppliers.xlsx").read_bytes()

    def test_generated_workbook_still_opens(self, tmp_path: Path) -> None:
        """Determinism must not have been bought by corrupting the file."""
        import pandas as pd

        write_demo_data(tmp_path, reference_date=date(2026, 1, 1))
        frame = pd.read_excel(tmp_path / "suppliers.xlsx", engine="openpyxl")
        assert frame.shape[0] == 120
        assert "supplier_name" in frame.columns

    def test_future_dates_stay_in_the_future(self, tmp_path: Path) -> None:
        """The seeded future dates must be relative to the reference date, not fixed.

        A hard-coded future date silently stops being a future date once that day
        arrives, and the future_date check would quietly go green on data that is
        still meant to be broken.
        """
        import pandas as pd

        reference = date(2030, 6, 1)
        write_demo_data(tmp_path, reference_date=reference)
        customers = pd.read_csv(tmp_path / "customers.csv", dtype=str)
        parsed = pd.to_datetime(customers["signup_date"], errors="coerce", format="mixed")
        assert (parsed > pd.Timestamp(reference)).any(), (
            "no signup_date is after the reference date"
        )

    def test_ground_truth_records_the_reference_date(self, tmp_path: Path) -> None:
        write_demo_data(tmp_path, reference_date=date(2026, 1, 1))
        truth = load_ground_truth(tmp_path / "ground_truth.json")
        assert truth["reference_date"] == "2026-01-01"

    def test_ground_truth_records_every_error(self, demo_dir: Path) -> None:
        truth = load_ground_truth(demo_dir / "ground_truth.json")
        assert truth["seeded_error_count"] == len(truth["seeded_errors"])
        assert truth["seeded_error_count"] > 100
        assert set(truth["datasets"]) == {"customers", "sales", "suppliers"}

    def test_every_seeded_issue_type_is_known(self, demo_dir: Path) -> None:
        truth = load_ground_truth(demo_dir / "ground_truth.json")
        for error in truth["seeded_errors"]:
            assert error["issue_type"] in ACCEPTABLE_DETECTORS, (
                f"seeded issue type {error['issue_type']} has no acceptable-detector entry"
            )


class TestDetectionCoverage:
    def test_every_seeded_defect_is_detected(
        self, demo_dir: Path, analyses: dict[str, object]
    ) -> None:
        """Every (dataset, column, issue type) that was seeded must be reported."""
        truth = load_ground_truth(demo_dir / "ground_truth.json")

        seeded: dict[str, set[tuple[str | None, str]]] = defaultdict(set)
        for error in truth["seeded_errors"]:
            seeded[error["dataset"]].add((error["column"], error["issue_type"]))

        undetected: list[str] = []
        for dataset, expected in seeded.items():
            findings = analyses[dataset].findings.findings  # type: ignore[attr-defined]
            for column, issue_type in sorted(expected, key=lambda item: (str(item[0]), item[1])):
                acceptable = ACCEPTABLE_DETECTORS[issue_type]
                if not any(
                    finding.issue_type in acceptable
                    and (finding.column == column or finding.column is None or column is None)
                    for finding in findings
                ):
                    undetected.append(f"{dataset}: {column} / {issue_type}")

        assert not undetected, "seeded defects that no check reported:\n" + "\n".join(undetected)

    @pytest.mark.parametrize(
        ("dataset", "issue_type"),
        [
            ("customers", IssueType.MISSING_VALUES),
            ("customers", IssueType.INVALID_EMAIL),
            ("customers", IssueType.INVALID_DATE),
            ("customers", IssueType.INCONSISTENT_DATE_FORMAT),
            ("customers", IssueType.FUTURE_DATE),
            ("customers", IssueType.EXACT_DUPLICATE_ROWS),
            ("customers", IssueType.PROBABLE_DUPLICATE_ROWS),
            ("customers", IssueType.NUMERIC_STORED_AS_TEXT),
            ("customers", IssueType.IMPOSSIBLE_NUMERIC),
            ("customers", IssueType.LEADING_TRAILING_WHITESPACE),
            ("customers", IssueType.INCONSISTENT_CAPITALIZATION),
            ("customers", IssueType.BUSINESS_RULE_VIOLATION),
            ("sales", IssueType.MISSING_VALUES),
            ("sales", IssueType.EXACT_DUPLICATE_ROWS),
            ("sales", IssueType.CONSTANT_COLUMN),
            ("sales", IssueType.BUSINESS_RULE_VIOLATION),
            ("sales", IssueType.INVALID_DATE),
            ("suppliers", IssueType.PROBABLE_DUPLICATE_ROWS),
            ("suppliers", IssueType.INVALID_EMAIL),
            ("suppliers", IssueType.MISSING_VALUES),
            ("suppliers", IssueType.BUSINESS_RULE_VIOLATION),
        ],
    )
    def test_headline_issue_types_fire(
        self, analyses: dict[str, object], dataset: str, issue_type: IssueType
    ) -> None:
        """The issue types shown in the demo must be present in each dataset."""
        found = {f.issue_type for f in analyses[dataset].findings.findings}  # type: ignore[attr-defined]
        assert issue_type in found


class TestAnalysisQuality:
    @pytest.mark.parametrize("dataset", ["customers", "sales", "suppliers"])
    def test_score_is_meaningfully_below_perfect(
        self, analyses: dict[str, object], dataset: str
    ) -> None:
        score = analyses[dataset].score  # type: ignore[attr-defined]
        assert 0 <= score.overall < 100, "deliberately dirty data should not score 100"

    @pytest.mark.parametrize("dataset", ["customers", "sales", "suppliers"])
    def test_every_finding_is_explained_and_attributed(
        self, analyses: dict[str, object], dataset: str
    ) -> None:
        for finding in analyses[dataset].findings.findings:  # type: ignore[attr-defined]
            assert len(finding.explanation) > 30
            assert finding.check_id
            assert 0.0 <= finding.confidence <= 1.0
            assert finding.affected_rows <= finding.row_count

    def test_clean_dataset_scores_high(self) -> None:
        """A dataset with no seeded errors must not be flooded with findings."""
        import pandas as pd

        clean = pd.DataFrame(
            {
                "customer_id": [f"CUST-{i:05d}" for i in range(50)],
                "full_name": [f"Person {i}" for i in range(50)],
                "email": [f"person{i}@example.com" for i in range(50)],
                "country": ["FR", "DE", "IT", "ES", "BE"] * 10,
                "signup_date": ["2024-01-15"] * 50,
                "revenue": [str(1000 + i) for i in range(50)],
            }
        )
        buffer = clean.to_csv(index=False).encode()
        settings = Settings(anthropic_api_key=None, rules_file=RULES_FILE)
        result = analyze_bytes(buffer, "clean.csv", settings=settings)

        assert result.score.overall >= 95, (
            "clean data should score highly; findings were: "
            + "; ".join(f.title for f in result.findings.findings)
        )
