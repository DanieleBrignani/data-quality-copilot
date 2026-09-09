"""A DuckDB-backed source that answers questions about a file without loading it.

The pandas pipeline reads a dataset into memory and asks Python. That is the right shape
for interactive review, and :mod:`scripts.benchmark` shows where it stops being so. Past
that point the answer is not a faster loop: it is to stop moving the data to the checks
and move the checks to the data.

This module is the smaller half of that idea. It opens a CSV or Parquet file as a
DuckDB view and runs aggregate SQL against it, so memory stays flat regardless of the
row count - the engine streams the file and returns counts.

**Column names are untrusted input.** They arrive from the header row of a file someone
uploaded, and they end up inside SQL text because no database lets you bind an
identifier as a parameter. Every identifier therefore goes through
:func:`quote_identifier`, and every *value* is bound rather than interpolated. Without
that, a header of ``evil" ; DROP TABLE t; --`` is an injection.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - keeps the import optional at runtime
    import duckdb

#: Formats DuckDB can read directly off disk.
SUPPORTED_SUFFIXES = frozenset({".csv", ".tsv", ".parquet"})


class SqlEngineError(RuntimeError):
    """The SQL engine could not open or query the source."""


def quote_identifier(name: str) -> str:
    """Return ``name`` as a SQL identifier that cannot break out of its quotes.

    Doubling embedded quotes is the standard escape. It is the only defence available
    here: an identifier cannot be a bound parameter, so the name is part of the query
    text whether we like it or not.
    """
    return '"' + name.replace('"', '""') + '"'


def numeric_expression(column: str) -> str:
    """Return SQL that reads ``column`` as a number, tolerating light formatting.

    The Python side parses currency symbols, thousands separators in either convention
    and parenthesised negatives. Reproducing all of that in SQL would be a second
    implementation of the same rules, free to disagree with the first. This covers the
    common Anglo formatting only - a bare number, optionally carrying a currency symbol
    or comma thousands separators - and anything else reads as NULL, which the checks
    treat as "not comparable" rather than "violates the rule".

    The divergence is measured rather than assumed: see
    ``tests/test_sql_engine.py::TestBothEnginesAgree``.
    """
    quoted = quote_identifier(column)
    cleaned = f"regexp_replace(trim({quoted}), '[€$£ ]', '', 'g')"
    return (
        f"COALESCE(TRY_CAST({cleaned} AS DOUBLE), TRY_CAST(replace({cleaned}, ',', '') AS DOUBLE))"
    )


def missing_expression(column: str) -> str:
    """Return SQL that is true when the cell holds no value.

    Matches the Python definition exactly: NULL, empty, or whitespace only.
    """
    quoted = quote_identifier(column)
    return f"({quoted} IS NULL OR trim({quoted}) = '')"


def normalised_expression(column: str) -> str:
    """Return SQL for the trimmed, case-folded value used when comparing rows."""
    return f"lower(trim({quote_identifier(column)}))"


@dataclass(slots=True)
class SqlSource:
    """A file exposed to SQL as a view, plus the facts needed to query it."""

    path: Path
    connection: duckdb.DuckDBPyConnection
    columns: tuple[str, ...]
    row_count: int
    view: str = "source"

    @property
    def table(self) -> str:
        """The view name, quoted for use in a query."""
        return quote_identifier(self.view)

    def query(self, sql: str, parameters: list[Any] | None = None) -> list[tuple[Any, ...]]:
        """Run a query against the view and return every row.

        Raises:
            SqlEngineError: If the query fails.
        """
        try:
            return list(self.connection.execute(sql, parameters or []).fetchall())
        except Exception as error:  # noqa: BLE001 - re-raised as the module's own type
            raise SqlEngineError(f"Query failed: {error}") from error

    def scalar(self, sql: str, parameters: list[Any] | None = None) -> Any:
        """Run a query expected to return a single value."""
        rows = self.query(sql, parameters)
        return rows[0][0] if rows and rows[0] else None

    def close(self) -> None:
        """Release the connection."""
        self.connection.close()

    def __enter__(self) -> SqlSource:
        """Return self, so the source can be used as a context manager."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Close the connection on the way out."""
        self.close()


def open_source(path: str | Path, view: str = "source") -> SqlSource:
    """Open a file as a SQL view.

    Everything is read as text, matching how the Python pipeline treats a CSV: a column
    is a string until something proves otherwise. Both engines then apply the same
    definition of "missing" and the same casts, which is what makes their results
    comparable.

    Args:
        path: The CSV, TSV or Parquet file.
        view: Name for the view inside the connection.

    Returns:
        An open :class:`SqlSource`. Close it, or use it as a context manager.

    Raises:
        SqlEngineError: If the file is missing, unsupported, or cannot be read.
    """
    source = Path(path)
    if not source.is_file():
        raise SqlEngineError(f"No such file: {source}")
    if source.suffix.lower() not in SUPPORTED_SUFFIXES:
        supported = ", ".join(sorted(SUPPORTED_SUFFIXES))
        raise SqlEngineError(f"The SQL engine reads {supported}, not '{source.suffix}'.")

    try:
        import duckdb
    except ImportError as error:  # pragma: no cover - dependency is declared
        raise SqlEngineError("duckdb is not installed; run `pip install duckdb`.") from error

    connection = duckdb.connect()
    try:
        # The path is passed through DuckDB's Python API rather than interpolated into
        # SQL text, so a filename can never become part of the statement.
        if source.suffix.lower() == ".parquet":
            relation = connection.read_parquet(str(source))
        else:
            relation = connection.read_csv(str(source), header=True, all_varchar=True)
        relation.create_view(view)

        quoted = quote_identifier(view)
        columns = tuple(row[0] for row in connection.execute(f"DESCRIBE {quoted}").fetchall())
        row_count = int(
            connection.execute(f"SELECT count(*) FROM {quoted}").fetchone()[0]  # type: ignore[index]
        )
    except Exception as error:  # noqa: BLE001 - re-raised as the module's own type
        connection.close()
        raise SqlEngineError(f"Could not read '{source.name}': {error}") from error

    return SqlSource(
        path=source,
        connection=connection,
        columns=columns,
        row_count=row_count,
        view=view,
    )
