from __future__ import annotations

import io

import pandas as pd
import pytest

from dqcopilot.config import Settings
from dqcopilot.ingestion import (
    CorruptFileError,
    DatasetTooLargeError,
    EmptyFileError,
    FileTooLargeError,
    UnsupportedFileTypeError,
    load_tabular,
    sanitize_filename,
    validate_extension,
)
from dqcopilot.models import IssueType
from dqcopilot.profiling.type_inference import missing_mask
from dqcopilot.services.analysis import analyze_bytes


class TestSanitizeFilename:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("customers.csv", "customers.csv"),
            ("../../etc/passwd.csv", "passwd.csv"),
            (r"C:\Users\bob\Desktop\sales.xlsx", "sales.xlsx"),
            ("my file (final).csv", "my_file_final.csv"),
            ("__weird__.csv", "weird.csv"),
            ("....csv", "csv"),
            ("", "upload"),
            ("   ", "upload"),
            ("données clients.csv", "donn_es_clients.csv"),
        ],
    )
    def test_produces_safe_names(self, raw: str, expected: str) -> None:
        assert sanitize_filename(raw) == expected

    def test_truncates_very_long_names(self) -> None:
        result = sanitize_filename("a" * 500 + ".csv")
        assert len(result) <= 130
        assert result.endswith(".csv")


class TestValidateExtension:
    @pytest.mark.parametrize("name", ["a.csv", "a.CSV", "a.xlsx", "a.XLSX"])
    def test_accepts_supported(self, name: str) -> None:
        assert validate_extension(name) in {".csv", ".xlsx"}

    @pytest.mark.parametrize("name", ["a.xls", "a.xlsm", "a.json", "a.parquet", "a.exe", "a"])
    def test_rejects_unsupported(self, name: str) -> None:
        with pytest.raises(UnsupportedFileTypeError):
            validate_extension(name)


class TestLoadTabular:
    def test_loads_csv(self, csv_bytes: bytes, settings: Settings) -> None:
        dataset = load_tabular(csv_bytes, "customers.csv", settings=settings)
        assert dataset.row_count == 6
        assert dataset.column_count == 5
        assert dataset.delimiter == ","
        assert dataset.extension == ".csv"

    def test_loads_xlsx(self, xlsx_bytes: bytes, settings: Settings) -> None:
        dataset = load_tabular(xlsx_bytes, "customers.xlsx", settings=settings)
        assert dataset.row_count == 6
        assert dataset.extension == ".xlsx"
        assert dataset.sheet_name

    def test_csv_keeps_values_as_raw_text(self, settings: Settings) -> None:
        content = b'amount\n007\n"1,234.50"\n'
        dataset = load_tabular(content, "a.csv", settings=settings)
        # Leading zeros and thousands separators must survive ingestion.
        assert dataset.frame["amount"].tolist() == ["007", "1,234.50"]

    def test_single_column_file_is_not_split_on_data_commas(self, settings: Settings) -> None:
        """A comma inside a quoted value must not turn one column into two."""
        content = b'amount\n"1,234.50"\n"2,000.00"\n'
        dataset = load_tabular(content, "a.csv", settings=settings)
        assert dataset.delimiter == ","
        assert dataset.column_count == 1

    def test_detects_semicolon_delimiter(self, settings: Settings) -> None:
        content = b"name;city\nAlice;Paris\nBob;Lyon\n"
        dataset = load_tabular(content, "a.csv", settings=settings)
        assert dataset.delimiter == ";"
        assert list(dataset.frame.columns) == ["name", "city"]

    def test_decodes_latin1_when_utf8_fails(self, settings: Settings) -> None:
        content = "name\nCafé\n".encode("cp1252")
        dataset = load_tabular(content, "a.csv", settings=settings)
        assert dataset.encoding in {"cp1252", "latin-1"}
        assert dataset.frame.shape[0] == 1

    def test_strips_and_deduplicates_headers(self, settings: Settings) -> None:
        content = b" name ,name,\nA,B,C\n"
        dataset = load_tabular(content, "a.csv", settings=settings)
        assert list(dataset.frame.columns) == ["name", "name_1", "column_3"]
        assert dataset.notes

    # ------------------------------------------------------------------ rejections

    def test_rejects_empty_file(self, settings: Settings) -> None:
        with pytest.raises(EmptyFileError):
            load_tabular(b"", "a.csv", settings=settings)

    def test_rejects_header_only_file(self, settings: Settings) -> None:
        with pytest.raises(EmptyFileError):
            load_tabular(b"a,b,c\n", "a.csv", settings=settings)

    def test_rejects_oversized_file(self, settings: Settings) -> None:
        payload = b"a\n" + b"1\n" * (settings.max_upload_bytes)
        with pytest.raises(FileTooLargeError):
            load_tabular(payload, "a.csv", settings=settings)

    def test_rejects_unsupported_extension(self, settings: Settings) -> None:
        with pytest.raises(UnsupportedFileTypeError):
            load_tabular(b"{}", "a.json", settings=settings)

    def test_rejects_binary_disguised_as_csv(self, settings: Settings) -> None:
        with pytest.raises(UnsupportedFileTypeError):
            load_tabular(b"MZ\x00\x00\x90\x00binary", "payload.csv", settings=settings)

    def test_rejects_zip_disguised_as_csv(self, settings: Settings) -> None:
        with pytest.raises(UnsupportedFileTypeError):
            load_tabular(b"PK\x03\x04rest", "archive.csv", settings=settings)

    def test_rejects_legacy_xls_renamed_to_xlsx(self, settings: Settings) -> None:
        content = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1" + b"\x00" * 32
        with pytest.raises(UnsupportedFileTypeError, match="legacy"):
            load_tabular(content, "old.xlsx", settings=settings)

    def test_rejects_text_renamed_to_xlsx(self, settings: Settings) -> None:
        with pytest.raises(UnsupportedFileTypeError, match="ZIP"):
            load_tabular(b"a,b\n1,2\n", "fake.xlsx", settings=settings)

    def test_rejects_corrupt_xlsx(self, settings: Settings) -> None:
        content = b"PK\x03\x04" + b"\x00" * 200
        with pytest.raises(CorruptFileError):
            load_tabular(content, "broken.xlsx", settings=settings)

    def test_rejects_ragged_csv(self, settings: Settings) -> None:
        with pytest.raises(CorruptFileError):
            load_tabular(b'a,b\n"unbalanced,2\n1,2,3,4,5\n', "a.csv", settings=settings)

    def test_rejects_too_many_rows(self) -> None:
        tiny = Settings(max_rows=3, anthropic_api_key=None)
        frame = pd.DataFrame({"a": range(10)})
        buffer = io.StringIO()
        frame.to_csv(buffer, index=False)
        with pytest.raises(DatasetTooLargeError):
            load_tabular(buffer.getvalue().encode(), "a.csv", settings=tiny)

    def test_rejects_too_many_columns(self) -> None:
        tiny = Settings(max_columns=2, anthropic_api_key=None)
        content = b"a,b,c\n1,2,3\n"
        with pytest.raises(DatasetTooLargeError):
            load_tabular(content, "a.csv", settings=tiny)

    def test_csv_formula_payload_stays_inert_text(self, settings: Settings) -> None:
        """A formula-looking CSV value is data, never something to evaluate."""
        content = b"payload\n=1+1\n=cmd|'/c calc'!A1\n@SUM(1+1)\n"
        dataset = load_tabular(content, "f.csv", settings=settings)
        assert dataset.frame["payload"].tolist() == ["=1+1", "=cmd|'/c calc'!A1", "@SUM(1+1)"]

    def test_xlsx_formulas_are_not_evaluated(self, settings: Settings) -> None:
        """Real formula cells are read through openpyxl's cached-value mode.

        A workbook written without cached results therefore yields empty cells - the
        formula is never computed - which the loader reports as an empty dataset.
        """
        from openpyxl import Workbook

        workbook = Workbook()
        sheet = workbook.active
        sheet.append(["payload"])
        sheet.append(["=1+1"])
        buffer = io.BytesIO()
        workbook.save(buffer)

        with pytest.raises(EmptyFileError):
            load_tabular(buffer.getvalue(), "f.xlsx", settings=settings)


class TestTheReaderDoesNotRewriteWhatItReads:
    """Reading a file must not edit it.

    Pandas converts about twenty tokens to "missing" while parsing, and this project
    used to add its own on top: ``unknown``, ``missing``, ``nil``, ``--``, ``?``. Each
    of those is a value somebody typed into a cell. Blanking it during the read is an
    unapproved edit that never reaches the audit log, disappears from the exported file,
    and hides the defect from the check written to report it.
    """

    @pytest.mark.parametrize(
        "written",
        ["unknown", "UNKNOWN", "missing", "nil", "--", "?", "n.a.", "N/A", "NA", "null", "None"],
    )
    def test_a_word_somebody_typed_survives_the_read(
        self, written: str, settings: Settings
    ) -> None:
        content = f"customer,country\nAlice,IT\nBruno,{written}\n".encode()

        dataset = load_tabular(content, "f.csv", settings=settings)

        assert dataset.frame["country"].tolist() == ["IT", written]

    def test_a_genuinely_empty_cell_is_still_missing(self, settings: Settings) -> None:
        content = b"customer,country\nAlice,IT\nBruno,\nChloe,   \n"

        dataset = load_tabular(content, "f.csv", settings=settings)
        country = dataset.frame["country"]

        assert country[0] == "IT"
        assert missing_mask(country).tolist() == [False, True, True]

    def test_the_distinction_reaches_the_findings(self, settings: Settings) -> None:
        """One empty cell and two typed words are three different rows, not three gaps."""
        content = b"customer,country\nAlice,IT\nBruno,unknown\nChloe,\nDavid,unknown\nElena,FR\n"

        result = analyze_bytes(content, "f.csv", settings=settings)
        by_type = {finding.issue_type: finding for finding in result.findings}

        assert by_type[IssueType.MISSING_VALUES].affected_rows == 1
        assert by_type[IssueType.PLACEHOLDER_VALUE].affected_rows == 2
