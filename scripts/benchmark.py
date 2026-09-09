"""Measure how the pipeline scales, and under what conditions it was measured.

Usage::

    python scripts/benchmark.py
    python scripts/benchmark.py --rows 10000,100000 --repeats 5

The point is not a marketing number. It is to know where this design stops being the
right one, so the answer to "does it handle large data?" is a measurement rather than a
shrug.

**The environment is printed with the results on purpose.** These timings were first
taken on a laptop with 2 GB of free memory, a sync client and an antivirus running, and
they came out three times slower than the same code on the same machine an hour earlier
- including for code that had not changed. A benchmark that does not say what else was
running is not a measurement, it is an anecdote. The script warns when free memory is
low, because that is the condition that quietly invalidates everything below it.
"""

from __future__ import annotations

import argparse
import platform
import statistics
import sys
import time
import tracemalloc
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from dqcopilot.profiling import profile_dataset
from dqcopilot.validation import CheckContext, run_checks

DEFAULT_ROWS = (10_000, 50_000, 200_000)
DEFAULT_REPEATS = 3

#: Below this much free memory the machine is likely paging and the timings are noise.
LOW_MEMORY_GB = 4.0


@dataclass(frozen=True, slots=True)
class Measurement:
    """One row of the results table."""

    rows: int
    profile_seconds: float
    checks_seconds: float
    peak_mb: float
    findings: int

    @property
    def total_seconds(self) -> float:
        """Wall time for profiling plus every check."""
        return self.profile_seconds + self.checks_seconds


def synthetic_frame(rows: int, seed: int = 42) -> pd.DataFrame:
    """Build a frame with the defect shapes the checks look for.

    Deliberately not clean data: a benchmark on a spotless dataset measures the fast
    path and reports a number the tool will never reach in use.
    """
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "customer_id": [f"C{index:08d}" for index in range(rows)],
            "full_name": rng.choice(
                ["Alice Rossi", "BRUNO DUPONT", " Chloe Weber ", "Marta Neri"], rows
            ),
            "email": rng.choice(["a@example.com", "malformed@", "c@example.com"], rows),
            "country": rng.choice(["FR", "fr", "DE", "IT"], rows),
            "segment": rng.choice(["Enterprise", "SMB", "N/A"], rows),
            "age": rng.integers(0, 130, rows),
            "signup_date": rng.choice(["2023-01-15", "15/03/2023", "not a date"], rows),
            "revenue": rng.normal(5000, 2000, rows).round(2),
        }
    )


def measure_once(frame: pd.DataFrame) -> Measurement:
    """Profile and check one frame, recording time and peak allocation."""
    tracemalloc.start()
    started = time.perf_counter()
    profile = profile_dataset(frame, "benchmark")
    profiled = time.perf_counter()
    findings = run_checks(CheckContext(frame=frame, profile=profile))
    finished = time.perf_counter()
    peak = tracemalloc.get_traced_memory()[1] / 1e6
    tracemalloc.stop()

    return Measurement(
        rows=len(frame),
        profile_seconds=profiled - started,
        checks_seconds=finished - profiled,
        peak_mb=peak,
        findings=len(findings),
    )


def _free_memory_gb() -> float | None:
    """Free physical memory in GB, or None when it cannot be determined."""
    try:
        import psutil
    except ImportError:
        return None
    return float(psutil.virtual_memory().available) / 1e9


def describe_environment() -> list[str]:
    """Return the facts a reader needs to judge whether the numbers mean anything."""
    lines = [
        f"python {platform.python_version()} · pandas {pd.__version__} · numpy {np.__version__}",
        f"{platform.system()} {platform.release()} · {platform.machine()}",
    ]
    free = _free_memory_gb()
    if free is None:
        lines.append("free memory: unknown (install psutil to record it)")
    else:
        lines.append(f"free memory at start: {free:.1f} GB")
        if free < LOW_MEMORY_GB:
            lines.append(
                f"WARNING: under {LOW_MEMORY_GB:.0f} GB free. The machine is probably paging "
                "and these timings say more about it than about the code."
            )
    return lines


def run(rows: tuple[int, ...], repeats: int) -> int:
    """Run the benchmark and print a table. Returns a process exit code."""
    for line in describe_environment():
        print(line)
    print()

    header = f"{'rows':>10} {'profile':>9} {'checks':>9} {'total':>9} {'peak MB':>9} {'spread':>8}"
    print(header)
    print("-" * len(header))

    for count in rows:
        frame = synthetic_frame(count)
        # The first pass warms import-time caches (date format tables, regexes); timing
        # it would charge the smallest dataset for everyone else's setup.
        measure_once(frame)
        results = [measure_once(frame) for _ in range(repeats)]

        totals = sorted(result.total_seconds for result in results)
        median = statistics.median(totals)
        spread = (totals[-1] - totals[0]) / median if median else 0.0
        best = min(results, key=lambda result: result.total_seconds)

        print(
            f"{count:>10,} {best.profile_seconds:>8.2f}s {best.checks_seconds:>8.2f}s "
            f"{median:>8.2f}s {best.peak_mb:>9.0f} {spread:>7.0%}"
        )

    print(
        "\nTimes are the median of the repeats; profile/checks come from the fastest run. "
        "A spread above ~20% means the machine was busy and the run is not comparable."
    )
    return 0


def compare_engines(rows: int, workdir: Path) -> int:
    """Run both engines over the same file and print what each costs.

    The comparison that matters is not seconds but peak memory: pandas holds the dataset,
    DuckDB streams it. One of those numbers grows with the file and the other does not.
    """
    from dqcopilot.config import get_settings
    from dqcopilot.rules.loader import load_rules_or_none
    from dqcopilot.sql.checks import run_sql_checks
    from dqcopilot.sql.engine import open_source

    path = workdir / f"benchmark-{rows}.csv"
    print(f"writing {rows:,} rows to {path} ...")
    synthetic_frame(rows).to_csv(path, index=False)
    print(f"file on disk: {path.stat().st_size / 1e6:.0f} MB\n")

    rules = load_rules_or_none(get_settings().rules_file)

    tracemalloc.start()
    started = time.perf_counter()
    frame = pd.read_csv(path, dtype=str)
    profile = profile_dataset(frame, "benchmark")
    python_findings = run_checks(CheckContext(frame=frame, profile=profile, rules=rules))
    python_seconds = time.perf_counter() - started
    python_peak = tracemalloc.get_traced_memory()[1] / 1e6
    tracemalloc.stop()
    del frame

    tracemalloc.start()
    started = time.perf_counter()
    with open_source(path) as source:
        sql_findings = run_sql_checks(source, rules)
    sql_seconds = time.perf_counter() - started
    sql_peak = tracemalloc.get_traced_memory()[1] / 1e6
    tracemalloc.stop()

    print(f"{'engine':<10} {'time':>9} {'peak MB':>9} {'findings':>9}")
    print("-" * 40)
    print(f"{'pandas':<10} {python_seconds:>8.1f}s {python_peak:>9.0f} {len(python_findings):>9}")
    print(f"{'duckdb':<10} {sql_seconds:>8.1f}s {sql_peak:>9.0f} {len(sql_findings):>9}")
    print(
        "\nThe finding counts are not meant to match: the SQL engine implements the four "
        "checks SQL expresses well, not all nineteen. Compare the peak memory."
    )
    path.unlink(missing_ok=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rows",
        default=",".join(str(value) for value in DEFAULT_ROWS),
        help="Comma-separated row counts to measure.",
    )
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument(
        "--compare-engines",
        type=int,
        metavar="ROWS",
        help="Write a CSV of ROWS rows and measure pandas against DuckDB on it.",
    )
    parser.add_argument(
        "--workdir",
        type=Path,
        default=Path("."),
        help="Where to write the temporary CSV for --compare-engines.",
    )
    args = parser.parse_args()

    try:
        rows = tuple(int(value) for value in args.rows.split(","))
    except ValueError:
        print(
            f"--rows must be a comma-separated list of integers, got {args.rows!r}",
            file=sys.stderr,
        )
        return 2

    if args.compare_engines:
        for line in describe_environment():
            print(line)
        print()
        return compare_engines(args.compare_engines, args.workdir)

    return run(rows, max(1, args.repeats))


if __name__ == "__main__":
    raise SystemExit(main())
