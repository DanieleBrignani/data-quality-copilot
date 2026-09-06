"""Reproducible synthetic demo datasets with a machine-readable ground truth.

Every dataset is generated from a fixed seed and every error is injected through
:meth:`_ErrorLog.record`, so the accompanying ``ground_truth.json`` always lists exactly
what was broken, in which row and in which column. That file is what makes the detection
rate measurable instead of anecdotal.

**All data here is fabricated.** Names, companies, emails and identifiers are generated
from fixed word lists and resolve to nothing real.
"""

from __future__ import annotations

import io
import json
import random
import re
import zipfile
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from dqcopilot.models.enums import IssueType

SEED = 20240601

FIRST_NAMES = [
    "Alice",
    "Bruno",
    "Chloé",
    "David",
    "Elena",
    "François",
    "Greta",
    "Hugo",
    "Ingrid",
    "Jonas",
    "Katarina",
    "Luca",
    "Marta",
    "Nils",
    "Olivia",
    "Pablo",
    "Quentin",
    "Rosa",
    "Sven",
    "Teresa",
    "Ulrich",
    "Valentina",
    "Wouter",
    "Yara",
]
LAST_NAMES = [
    "Martin",
    "Dupont",
    "Rossi",
    "Fischer",
    "Silva",
    "Novak",
    "Andersen",
    "Kowalski",
    "Moreau",
    "Weber",
    "Costa",
    "Jensen",
    "Bakker",
    "Lambert",
    "Esposito",
    "Nilsson",
]
COMPANY_STEMS = [
    "Northwind",
    "Globex",
    "Initech",
    "Umbrella",
    "Soylent",
    "Vandelay",
    "Hooli",
    "Wonka",
    "Stark",
    "Wayne",
    "Acme",
    "Cyberdyne",
    "Tyrell",
    "Gringotts",
]
COMPANY_SUFFIXES = ["Ltd", "GmbH", "S.p.A.", "SAS", "BV", "AB", "SL", "Oy"]
COUNTRIES = ["FR", "DE", "IT", "ES", "BE", "NL", "PT", "SE"]
SEGMENTS = ["Enterprise", "Mid-Market", "SMB", "Public Sector"]
STATUSES = ["pending", "confirmed", "shipped", "delivered", "cancelled"]
CATEGORIES = ["Hardware", "Software", "Services", "Consumables"]

#: Deliberately wrong variants injected to exercise the category checks.
COUNTRY_VARIANTS = {
    "FR": ["fr", " FR", "France", "FRA", "fr "],
    "DE": ["de", "Germany", "DEU", " de"],
    "IT": ["it", "Italy", "ITA"],
    "ES": ["es", "Spain"],
}
NUMBER_WORDS = ["twelve", "one thousand", "n/d", "TBD", "approx 500"]


@dataclass
class SeededError:
    """One deliberately injected error, recorded for the ground truth file."""

    dataset: str
    row: int
    column: str | None
    issue_type: str
    description: str
    original_value: str | None = None
    injected_value: str | None = None


@dataclass
class _ErrorLog:
    """Collects the errors injected into one dataset."""

    dataset: str
    errors: list[SeededError] = field(default_factory=list)

    def record(
        self,
        row: int,
        column: str | None,
        issue_type: IssueType,
        description: str,
        original: Any = None,
        injected: Any = None,
    ) -> None:
        """Record one injected error."""
        self.errors.append(
            SeededError(
                dataset=self.dataset,
                row=row,
                column=column,
                issue_type=issue_type.value,
                description=description,
                original_value=None if original is None else str(original),
                injected_value=None if injected is None else str(injected),
            )
        )


def _rng() -> random.Random:
    return random.Random(SEED)


def _email(first: str, last: str, domain: str = "example.com") -> str:
    ascii_first = first.lower().translate(
        str.maketrans("áàâäéèêëíìîïóòôöúùûüç", "aaaaeeeeiiiioooouuuuc")
    )
    ascii_last = last.lower().translate(
        str.maketrans("áàâäéèêëíìîïóòôöúùûüç", "aaaaeeeeiiiioooouuuuc")
    )
    return f"{ascii_first}.{ascii_last}@{domain}"


# --------------------------------------------------------------------------- customers


def build_customers(rows: int = 220) -> tuple[pd.DataFrame, list[SeededError]]:
    """Build the synthetic customer dataset."""
    rng = _rng()
    log = _ErrorLog("customers")
    records: list[dict[str, Any]] = []

    base_date = date(2022, 1, 1)
    for index in range(rows):
        first = rng.choice(FIRST_NAMES)
        last = rng.choice(LAST_NAMES)
        signup = base_date + timedelta(days=rng.randint(0, 900))
        records.append(
            {
                "customer_id": f"CUST-{index + 1:05d}",
                "full_name": f"{first} {last}",
                "email": _email(first, last),
                "country": rng.choice(COUNTRIES),
                "segment": rng.choice(SEGMENTS),
                "age": rng.randint(21, 78),
                "signup_date": signup.isoformat(),
                "revenue": round(rng.uniform(500, 90_000), 2),
                "is_active": rng.choice(["true", "false"]),
            }
        )

    frame = pd.DataFrame(records)
    _inject_customer_errors(frame, log, rng)
    return frame, log.errors


def _inject_customer_errors(frame: pd.DataFrame, log: _ErrorLog, rng: random.Random) -> None:
    """Break the customer dataset in known, recorded ways."""
    # 1. Missing values
    for row in [3, 17, 42, 88, 130]:
        log.record(row, "email", IssueType.MISSING_VALUES, "Email removed", frame.at[row, "email"])
        frame.at[row, "email"] = None
    for row in [11, 59, 121]:
        log.record(
            row, "country", IssueType.MISSING_VALUES, "Country blanked", frame.at[row, "country"]
        )
        frame.at[row, "country"] = "   "
    for row in [7, 64]:
        log.record(row, "segment", IssueType.MISSING_VALUES, "Segment set to placeholder")
        frame.at[row, "segment"] = "unknown"

    # 2. Malformed emails
    broken_emails = {
        5: "alice.martin@",
        23: "bruno.dupont.example.com",
        61: "chloe@@example.com",
        95: "david.weber@example",
        140: "elena rossi@example.com",
    }
    for row, value in broken_emails.items():
        log.record(
            row, "email", IssueType.INVALID_EMAIL, "Malformed email", frame.at[row, "email"], value
        )
        frame.at[row, "email"] = value

    # 3. Inconsistent countries
    for row in [2, 14, 33, 47, 70, 91, 108, 155, 176, 199]:
        original = str(frame.at[row, "country"])
        variants = COUNTRY_VARIANTS.get(original)
        if not variants:
            continue
        value = rng.choice(variants)
        log.record(
            row, "country", IssueType.INCONSISTENT_CATEGORY, "Country variant", original, value
        )
        frame.at[row, "country"] = value

    # 4. Whitespace and capitalisation on names
    for row in [1, 26, 55, 102, 148]:
        original = str(frame.at[row, "full_name"])
        value = f"  {original} "
        log.record(
            row, "full_name", IssueType.LEADING_TRAILING_WHITESPACE, "Padded name", original, value
        )
        frame.at[row, "full_name"] = value
    for row in [9, 38, 77, 119, 163]:
        original = str(frame.at[row, "full_name"])
        value = original.upper()
        log.record(
            row,
            "full_name",
            IssueType.INCONSISTENT_CAPITALIZATION,
            "Upper-cased name",
            original,
            value,
        )
        frame.at[row, "full_name"] = value

    # 5. Impossible ages
    for row, value in {12: 0, 44: 199, 83: -5, 167: 150}.items():
        log.record(
            row,
            "age",
            IssueType.IMPOSSIBLE_NUMERIC,
            "Age outside 0-120",
            frame.at[row, "age"],
            value,
        )
        frame.at[row, "age"] = value

    # 6. Negative revenue
    for row in [8, 51, 96, 134]:
        original = frame.at[row, "revenue"]
        value = -abs(float(original))
        log.record(
            row, "revenue", IssueType.IMPOSSIBLE_NUMERIC, "Revenue made negative", original, value
        )
        frame.at[row, "revenue"] = value

    # 7. Revenue as decorated text (forces the whole column to text)
    frame["revenue"] = frame["revenue"].map(lambda value: f"{value:,.2f}")
    for row in [20, 73, 110]:
        original = frame.at[row, "revenue"]
        value = f"€ {original}"
        log.record(
            row,
            "revenue",
            IssueType.NUMERIC_STORED_AS_TEXT,
            "Currency symbol added",
            original,
            value,
        )
        frame.at[row, "revenue"] = value
    for row in [35, 128]:
        original = frame.at[row, "revenue"]
        value = rng.choice(NUMBER_WORDS)
        log.record(
            row, "revenue", IssueType.MIXED_DATATYPES, "Number written as words", original, value
        )
        frame.at[row, "revenue"] = value
    log.record(
        -1, "revenue", IssueType.NUMERIC_STORED_AS_TEXT, "Whole revenue column stored as text"
    )

    # 8. Dates: mixed formats, impossible days, future dates
    for row in [4, 29, 66, 113, 158]:
        original = str(frame.at[row, "signup_date"])
        parsed = date.fromisoformat(original)
        value = parsed.strftime("%d/%m/%Y")
        log.record(
            row,
            "signup_date",
            IssueType.INCONSISTENT_DATE_FORMAT,
            "Reformatted to DD/MM/YYYY",
            original,
            value,
        )
        frame.at[row, "signup_date"] = value
    for row, value in {16: "2023-02-30", 87: "2024-13-01", 145: "not a date"}.items():
        log.record(
            row,
            "signup_date",
            IssueType.INVALID_DATE,
            "Impossible date",
            frame.at[row, "signup_date"],
            value,
        )
        frame.at[row, "signup_date"] = value
    future = (date.today() + timedelta(days=400)).isoformat()
    for row in [31, 104]:
        log.record(
            row,
            "signup_date",
            IssueType.FUTURE_DATE,
            "Date moved to the future",
            frame.at[row, "signup_date"],
            future,
        )
        frame.at[row, "signup_date"] = future

    # 9. Duplicates - exact and near
    for source, target in [(10, 200), (25, 201)]:
        log.record(target, None, IssueType.EXACT_DUPLICATE_ROWS, f"Exact copy of row {source}")
        frame.loc[target] = frame.loc[source].copy()
    # Near duplicates: same person, different formatting, but at least one field
    # genuinely differs so the row is NOT an exact duplicate after normalisation.
    for source, target in [(40, 202), (57, 203)]:
        frame.loc[target] = frame.loc[source].copy()
        original = str(frame.at[target, "full_name"])
        variant = original.strip().lower()
        frame.at[target, "full_name"] = variant
        frame.at[target, "customer_id"] = frame.at[source, "customer_id"]
        # Differing payload keeps this a *near* duplicate rather than an exact one.
        frame.at[target, "age"] = int(frame.at[source, "age"]) + 1
        frame.at[target, "segment"] = (
            "SMB" if frame.at[source, "segment"] != "SMB" else "Enterprise"
        )
        log.record(
            target,
            None,
            IssueType.PROBABLE_DUPLICATE_ROWS,
            f"Same customer as row {source}, re-entered with different formatting",
            original,
            variant,
        )


# --------------------------------------------------------------------------- sales


def build_sales(rows: int = 300) -> tuple[pd.DataFrame, list[SeededError]]:
    """Build the synthetic sales dataset."""
    rng = _rng()
    log = _ErrorLog("sales")
    records: list[dict[str, Any]] = []

    base_date = date(2023, 1, 1)
    for index in range(rows):
        order_date = base_date + timedelta(days=rng.randint(0, 700))
        quantity = rng.randint(1, 40)
        unit_price = round(rng.uniform(9.9, 899.0), 2)
        records.append(
            {
                "order_id": f"ORD-{index + 1:06d}",
                "customer_id": f"CUST-{rng.randint(1, 220):05d}",
                "order_date": order_date.isoformat(),
                "delivery_date": (order_date + timedelta(days=rng.randint(1, 21))).isoformat(),
                "product_category": rng.choice(CATEGORIES),
                "quantity": quantity,
                "unit_price": unit_price,
                "discount_percent": rng.choice([0, 0, 0, 5, 10, 15, 20]),
                "total_amount": round(quantity * unit_price, 2),
                "status": rng.choice(STATUSES),
                "currency": "EUR",
            }
        )

    frame = pd.DataFrame(records)
    _inject_sales_errors(frame, log, rng)
    return frame, log.errors


def _inject_sales_errors(frame: pd.DataFrame, log: _ErrorLog, rng: random.Random) -> None:
    """Break the sales dataset in known, recorded ways."""
    # Missing values
    for row in [6, 48, 133, 210, 266]:
        log.record(
            row,
            "total_amount",
            IssueType.MISSING_VALUES,
            "Amount removed",
            frame.at[row, "total_amount"],
        )
        frame.at[row, "total_amount"] = None
    for row in [19, 155]:
        log.record(
            row,
            "customer_id",
            IssueType.MISSING_VALUES,
            "Customer link removed",
            frame.at[row, "customer_id"],
        )
        frame.at[row, "customer_id"] = None

    # Negative amounts and impossible quantities
    for row in [13, 72, 189, 245]:
        original = frame.at[row, "total_amount"]
        value = -abs(float(original)) if pd.notna(original) else -100.0
        log.record(
            row,
            "total_amount",
            IssueType.IMPOSSIBLE_NUMERIC,
            "Amount made negative",
            original,
            value,
        )
        frame.at[row, "total_amount"] = value
    for row, value in {27: 0, 91: -3, 174: 99999}.items():
        log.record(
            row,
            "quantity",
            IssueType.BUSINESS_RULE_VIOLATION,
            "Quantity outside 1-10000",
            frame.at[row, "quantity"],
            value,
        )
        frame.at[row, "quantity"] = value
    for row, value in {36: 120, 203: -10}.items():
        log.record(
            row,
            "discount_percent",
            IssueType.BUSINESS_RULE_VIOLATION,
            "Discount outside 0-100",
            frame.at[row, "discount_percent"],
            value,
        )
        frame.at[row, "discount_percent"] = value

    # Amounts written with mixed conventions -> forces the column to text
    frame["total_amount"] = frame["total_amount"].map(
        lambda value: "" if pd.isna(value) else f"{value:,.2f}"
    )
    for row in [22, 118, 231]:
        original = frame.at[row, "total_amount"]
        value = f"${original}"
        log.record(
            row,
            "total_amount",
            IssueType.NUMERIC_STORED_AS_TEXT,
            "Dollar sign added",
            original,
            value,
        )
        frame.at[row, "total_amount"] = value
    for row in [55, 198]:
        original = str(frame.at[row, "total_amount"])
        value = original.replace(",", "X").replace(".", ",").replace("X", ".")
        log.record(
            row,
            "total_amount",
            IssueType.NUMERIC_STORED_AS_TEXT,
            "European separators",
            original,
            value,
        )
        frame.at[row, "total_amount"] = value

    # Dates
    for row in [9, 84, 167, 250]:
        original = str(frame.at[row, "order_date"])
        value = date.fromisoformat(original).strftime("%m/%d/%Y")
        log.record(
            row,
            "order_date",
            IssueType.INCONSISTENT_DATE_FORMAT,
            "Reformatted to MM/DD/YYYY",
            original,
            value,
        )
        frame.at[row, "order_date"] = value
    for row, value in {41: "2023-06-31", 122: "31/12/2023 25:00"}.items():
        log.record(
            row,
            "order_date",
            IssueType.INVALID_DATE,
            "Impossible date",
            frame.at[row, "order_date"],
            value,
        )
        frame.at[row, "order_date"] = value
    future = (date.today() + timedelta(days=200)).isoformat()
    for row in [63, 209]:
        log.record(
            row,
            "order_date",
            IssueType.FUTURE_DATE,
            "Order date in the future",
            frame.at[row, "order_date"],
            future,
        )
        frame.at[row, "order_date"] = future
    for row in [77, 181]:
        original = str(frame.at[row, "delivery_date"])
        value = (
            date.fromisoformat(str(frame.at[row, "order_date"])) - timedelta(days=5)
        ).isoformat()
        log.record(
            row,
            "delivery_date",
            IssueType.BUSINESS_RULE_VIOLATION,
            "Delivery before order",
            original,
            value,
        )
        frame.at[row, "delivery_date"] = value

    # Categories and status
    for row in [15, 68, 143, 220]:
        original = str(frame.at[row, "product_category"])
        value = original.lower()
        log.record(
            row,
            "product_category",
            IssueType.INCONSISTENT_CAPITALIZATION,
            "Lower-cased category",
            original,
            value,
        )
        frame.at[row, "product_category"] = value
    for row in [30, 159]:
        original = str(frame.at[row, "status"])
        value = original.upper()
        log.record(
            row,
            "status",
            IssueType.INCONSISTENT_CAPITALIZATION,
            "Upper-cased status",
            original,
            value,
        )
        frame.at[row, "status"] = value
    for row, value in {52: "in transit", 212: "COMPLETE"}.items():
        log.record(
            row,
            "status",
            IssueType.BUSINESS_RULE_VIOLATION,
            "Status outside the vocabulary",
            frame.at[row, "status"],
            value,
        )
        frame.at[row, "status"] = value

    # Duplicates
    for source, target in [(100, 290), (140, 291), (175, 292)]:
        log.record(target, None, IssueType.EXACT_DUPLICATE_ROWS, f"Exact copy of row {source}")
        frame.loc[target] = frame.loc[source].copy()

    # Constant column
    log.record(-1, "currency", IssueType.CONSTANT_COLUMN, "Currency is the same on every row")


# --------------------------------------------------------------------------- suppliers


def build_suppliers(rows: int = 120) -> tuple[pd.DataFrame, list[SeededError]]:
    """Build the synthetic supplier dataset."""
    rng = _rng()
    log = _ErrorLog("suppliers")
    records: list[dict[str, Any]] = []

    for index in range(rows):
        stem = rng.choice(COMPANY_STEMS)
        suffix = rng.choice(COMPANY_SUFFIXES)
        country = rng.choice(COUNTRIES)
        records.append(
            {
                "supplier_id": f"SUP-{index + 1:04d}",
                "supplier_name": f"{stem} {suffix}",
                "contact_email": f"contact@{stem.lower()}-{index + 1}.example.com",
                "country": country,
                "vat_number": f"{country}{rng.randint(10_000_000, 999_999_999)}",
                "onboarded_on": (
                    date(2019, 1, 1) + timedelta(days=rng.randint(0, 2000))
                ).isoformat(),
                "annual_spend": round(rng.uniform(5_000, 750_000), 2),
                "rating": rng.choice(["A", "B", "C", "D"]),
            }
        )

    frame = pd.DataFrame(records)
    _inject_supplier_errors(frame, log, rng)
    return frame, log.errors


def _inject_supplier_errors(frame: pd.DataFrame, log: _ErrorLog, rng: random.Random) -> None:
    """Break the supplier dataset, focusing on near-duplicate company names."""
    # The headline problem for suppliers: the same company entered several ways.
    # Each variant is derived from the row's *actual* generated name so the pair really
    # is the same company; the transformations are the ones seen in real vendor masters.
    variant_styles = [
        lambda name: name.upper().replace("LTD", "LIMITED"),
        lambda name: name.lower().replace(" ", "  "),
        lambda name: ".".join(name) if len(name) < 4 else name.replace(" ", " S.").rstrip("."),
        lambda name: f"  {name.casefold()} ",
        lambda name: name.replace(".", "").replace(" ", ""),
    ]
    target = 110
    for source, style in zip([4, 18, 37, 55, 71], variant_styles, strict=True):
        if target >= len(frame):
            break
        frame.loc[target] = frame.loc[source].copy()
        original = str(frame.at[source, "supplier_name"])
        variant = style(original)
        frame.at[target, "supplier_name"] = variant
        frame.at[target, "supplier_id"] = f"SUP-{target + 1:04d}"
        # Differing spend keeps this a *near* duplicate rather than an exact one.
        frame.at[target, "annual_spend"] = round(float(frame.at[source, "annual_spend"]) + 137.5, 2)
        log.record(
            target,
            None,
            IssueType.PROBABLE_DUPLICATE_ROWS,
            f"Same company as row {source}, written differently",
            original,
            variant,
        )
        target += 1

    # Missing values
    for row in [2, 29, 63, 98]:
        log.record(
            row,
            "contact_email",
            IssueType.MISSING_VALUES,
            "Contact email removed",
            frame.at[row, "contact_email"],
        )
        frame.at[row, "contact_email"] = None
    for row in [11, 77]:
        log.record(
            row,
            "vat_number",
            IssueType.MISSING_VALUES,
            "VAT number removed",
            frame.at[row, "vat_number"],
        )
        frame.at[row, "vat_number"] = ""

    # Malformed VAT numbers
    for row, value in {7: "123456789", 44: "FR-123", 86: "fr123456789"}.items():
        log.record(
            row,
            "vat_number",
            IssueType.BUSINESS_RULE_VIOLATION,
            "VAT number does not match the pattern",
            frame.at[row, "vat_number"],
            value,
        )
        frame.at[row, "vat_number"] = value

    # Malformed emails
    for row, value in {13: "contact@", 52: "no-at-sign.example.com"}.items():
        log.record(
            row,
            "contact_email",
            IssueType.INVALID_EMAIL,
            "Malformed email",
            frame.at[row, "contact_email"],
            value,
        )
        frame.at[row, "contact_email"] = value

    # Country variants and padding
    for row in [8, 34, 69, 92]:
        original = str(frame.at[row, "country"])
        value = rng.choice(COUNTRY_VARIANTS.get(original, [original.lower()]))
        log.record(
            row, "country", IssueType.INCONSISTENT_CATEGORY, "Country variant", original, value
        )
        frame.at[row, "country"] = value

    # Spend as text with mixed separators
    frame["annual_spend"] = frame["annual_spend"].map(lambda value: f"{value:,.2f}")
    for row in [5, 40, 81]:
        original = frame.at[row, "annual_spend"]
        value = f"£{original}"
        log.record(
            row,
            "annual_spend",
            IssueType.NUMERIC_STORED_AS_TEXT,
            "Pound sign added",
            original,
            value,
        )
        frame.at[row, "annual_spend"] = value

    # Future onboarding date
    future = (date.today() + timedelta(days=90)).isoformat()
    log.record(
        23,
        "onboarded_on",
        IssueType.FUTURE_DATE,
        "Onboarding date in the future",
        frame.at[23, "onboarded_on"],
        future,
    )
    frame.at[23, "onboarded_on"] = future

    # Impossible date
    log.record(
        58,
        "onboarded_on",
        IssueType.INVALID_DATE,
        "Impossible date",
        frame.at[58, "onboarded_on"],
        "2021-02-30",
    )
    frame.at[58, "onboarded_on"] = "2021-02-30"


# --------------------------------------------------------------------------- writing

DATASETS = {
    "customers": build_customers,
    "sales": build_sales,
    "suppliers": build_suppliers,
}


def generate_all() -> tuple[dict[str, pd.DataFrame], list[SeededError]]:
    """Build every demo dataset and the combined list of seeded errors."""
    frames: dict[str, pd.DataFrame] = {}
    errors: list[SeededError] = []
    for name, builder in DATASETS.items():
        frame, dataset_errors = builder()
        frames[name] = frame.reset_index(drop=True)
        errors.extend(dataset_errors)
    return frames, errors


#: Fixed document timestamp for the generated workbook. Naive on purpose: openpyxl
#: stores document properties as naive UTC, and a timezone-aware value is not applied.
XLSX_TIMESTAMP = datetime(2024, 1, 1, 0, 0, 0)  # noqa: DTZ001 - naive by design

#: Fixed (year, month, day, hour, minute, second) written into every ZIP entry.
XLSX_ZIP_DATE = (2024, 1, 1, 0, 0, 0)

#: openpyxl overwrites dcterms:modified with the current time inside save(), ignoring
#: whatever the properties were set to, so it has to be corrected in the written bytes.
_MODIFIED_RE = re.compile(rb"(<dcterms:modified[^>]*>)[^<]*(</dcterms:modified>)")
_FIXED_MODIFIED = b"2024-01-01T00:00:00Z"


def _write_reproducible_xlsx(frame: pd.DataFrame, path: Path) -> None:
    """Write a workbook whose bytes are identical on every run.

    Three separate sources of non-determinism have to be removed. The document
    properties in ``docProps/core.xml`` carry created/modified timestamps, which
    openpyxl fills with the current time. The XLSX container is also a ZIP, and every
    entry records the wall-clock time at which it was written - which openpyxl does not
    expose. The workbook is therefore built in memory and then repacked with fixed
    entry dates, in sorted order. openpyxl also rewrites ``dcterms:modified`` with
    the current time inside ``save()``, ignoring the value set on the properties, so
    that one field is corrected in the written bytes during the same pass.

    Without the repacking step, two runs in the same second happen to match and two runs
    a second apart do not, which is worse than plainly non-deterministic: it makes the
    build fail intermittently.
    """
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        frame.to_excel(writer, index=False, sheet_name="suppliers")
        properties = writer.book.properties
        properties.created = XLSX_TIMESTAMP
        properties.modified = XLSX_TIMESTAMP
        properties.creator = "dqcopilot demo data generator"
        properties.lastModifiedBy = "dqcopilot demo data generator"

    source = zipfile.ZipFile(io.BytesIO(buffer.getvalue()))
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as repacked:
        for name in sorted(source.namelist()):
            data = source.read(name)
            if name == "docProps/core.xml":
                data = _MODIFIED_RE.sub(rb"\g<1>" + _FIXED_MODIFIED + rb"\g<2>", data)
            info = zipfile.ZipInfo(name, date_time=XLSX_ZIP_DATE)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o600 << 16
            repacked.writestr(info, data)


def write_demo_data(output_dir: str | Path) -> Path:
    """Write every demo dataset plus ``ground_truth.json`` into ``output_dir``.

    Args:
        output_dir: Destination directory; created if it does not exist.

    Returns:
        The path of the ground truth file.
    """
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    frames, errors = generate_all()
    for name, frame in frames.items():
        frame.to_csv(destination / f"{name}.csv", index=False)

    # One dataset is also written as XLSX so the Excel path has a demo file too.
    _write_reproducible_xlsx(frames["suppliers"], destination / "suppliers.xlsx")

    ground_truth = {
        "seed": SEED,
        "generator_version": 1,
        "disclaimer": (
            "Fully synthetic data. Every name, company, email and identifier is "
            "fabricated and resolves to nothing real."
        ),
        "datasets": {
            name: {"rows": int(frame.shape[0]), "columns": list(map(str, frame.columns))}
            for name, frame in frames.items()
        },
        "seeded_errors": [asdict(error) for error in errors],
        "seeded_error_count": len(errors),
    }

    path = destination / "ground_truth.json"
    path.write_text(json.dumps(ground_truth, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def load_ground_truth(path: str | Path) -> dict[str, Any]:
    """Read a previously written ground truth file."""
    return json.loads(Path(path).read_text(encoding="utf-8"))
