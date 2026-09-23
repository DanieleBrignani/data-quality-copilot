"""What the checks say about data that has nothing wrong with it.

`test_demo_data_detection.py` measures the other direction: it asserts that every error
the generator injected is found. That measurement is precise and circular - it can only
find what someone thought to break, and a check that flagged every row would pass it.

This file is the counterweight, and it needs one distinction to be worth anything.

* An **asserted defect** claims something is wrong. On correct data that claim is false,
  and the count below must stay at zero.
* An **advisory finding** asks a human to look. Two people really can share a name, and
  no amount of code distinguishes that from one person entered twice - flagging it is
  the right behaviour, not a mistake. It still costs the reviewer attention, so it is
  counted separately rather than excused.

Every fixture here is data a careful person would call clean, chosen to be awkward for
one specific check.
"""

from __future__ import annotations

import pandas as pd
import pytest

from dqcopilot.profiling import profile_dataset
from dqcopilot.validation import CheckContext, run_checks

#: Checks whose findings ask for a decision rather than assert an error. Each one says so
#: in its own explanation: "an outlier is not necessarily wrong", "these need a human
#: decision", "worth confirming rather than fixing automatically".
ADVISORY_CHECKS = frozenset(
    {"suspicious_numeric", "constant_column", "probable_duplicate_rows", "business_rule"}
)


def ids(n: int) -> list[str]:
    return [str(i) for i in range(n)]


def homonyms() -> pd.DataFrame:
    """Forty distinct people, two of whom happen to share a name.

    The column is called `full_name` on purpose. An earlier version called it `person`,
    which is not a name the duplicate check recognises as identifying - so the check
    returned immediately and the fixture proved nothing while appearing to pass.
    """
    frame = pd.DataFrame({"full_name": [f"Persona {i}" for i in range(40)], "id": ids(40)})
    frame["full_name"] = frame["full_name"].mask(frame.index == 7, "Persona 3")
    return frame


#: name -> (frame, which check it is trying to trip)
CLEAN_FIXTURES: dict[str, tuple[pd.DataFrame, str]] = {
    "two-letter codes": (
        pd.DataFrame(
            {
                "state": ["CA", "NY", "ND", "TX", "FL", "ND", "CA", "TX", "NY", "FL"] * 3,
                "id": ids(30),
            }
        ),
        # 'ND' is North Dakota here, not an abbreviation of "not available".
        "placeholder_values",
    ),
    "categories that repeat because they are categories": (
        pd.DataFrame({"category": ["Hardware", "Software", "Services"] * 20, "id": ids(60)}),
        "placeholder_values",
    ),
    "two people with the same name": (homonyms(), "probable_duplicate_rows"),
    "correctly encoded accents": (
        pd.DataFrame(
            {
                "company": [
                    f"{name} {index}"
                    for index, name in enumerate(
                        ["Viganò", "Café", "São", "Frères", "Ångström", "Đặng"] * 6
                    )
                ],
                "id": ids(36),
            }
        ),
        "corrupted_encoding",
    ),
    "unusual but valid addresses": (
        pd.DataFrame(
            {
                "email": [f"nome+tag{i}@sotto.dominio.museum" for i in range(20)]
                + [f"x.y{i}@a-b.co.uk" for i in range(20)],
                "id": ids(40),
            }
        ),
        "invalid_email",
    ),
    "one consistent non-ISO date format": (
        pd.DataFrame(
            {"signup_date": [f"{i % 28 + 1:02d}/03/2024" for i in range(40)], "id": ids(40)}
        ),
        "inconsistent_date_format",
    ),
    "ordinary measurements": (
        pd.DataFrame(
            {
                "quantity": [str(1 + i % 50) for i in range(60)],
                "unit_price": [f"{10 + i % 30}.50" for i in range(60)],
            }
        ),
        "suspicious_numeric",
    ),
}


def findings_for(frame: pd.DataFrame) -> list:
    profile = profile_dataset(frame, "clean")
    return list(run_checks(CheckContext(frame=frame, profile=profile)))


class TestNothingIsAssertedAboutCleanData:
    @pytest.mark.parametrize("name", sorted(CLEAN_FIXTURES))
    def test_no_defect_is_claimed(self, name: str) -> None:
        frame, _ = CLEAN_FIXTURES[name]
        asserted = [f for f in findings_for(frame) if f.check_id not in ADVISORY_CHECKS]

        assert not asserted, "claimed on correct data: " + "; ".join(
            f"{f.check_id} ({f.affected_rows} rows)" for f in asserted
        )

    def test_the_advisory_cost_is_one_flag_across_the_whole_set(self) -> None:
        """Pinned, so a change that makes the checks noisier has to be deliberate.

        The single flag is the pair of homonyms, and it is correct: the tool cannot know
        whether that is two people or one record entered twice, so it asks.
        """
        advisory = [
            (name, f.check_id)
            for name, (frame, _) in CLEAN_FIXTURES.items()
            for f in findings_for(frame)
            if f.check_id in ADVISORY_CHECKS
        ]

        assert advisory == [("two people with the same name", "probable_duplicate_rows")]


class TestTheFixturesActuallyExerciseTheirCheck:
    """A fixture that never reaches its check proves nothing while appearing to pass.

    Two of these were written wrong the first time - one contained genuine duplicate
    rows, another named its column so that the duplicate check exited before running -
    and both looked like clean passes.
    """

    def test_the_duplicate_check_reaches_the_homonyms(self) -> None:
        from dqcopilot.validation.checks.duplicates import ProbableDuplicateRowsCheck

        frame = homonyms()
        context = CheckContext(frame=frame, profile=profile_dataset(frame, "clean"))

        assert ProbableDuplicateRowsCheck()._key_columns(context) == ["full_name"]

    @pytest.mark.parametrize("name", sorted(CLEAN_FIXTURES))
    def test_the_fixture_holds_no_exact_duplicates(self, name: str) -> None:
        """The first version of two fixtures did, which is why they 'found' defects."""
        frame, _ = CLEAN_FIXTURES[name]
        normalised = frame.apply(lambda col: col.astype("string").str.strip().str.casefold())

        assert not normalised.duplicated().any()
