"""Detect when one column's value is decided by another column's.

A neighbourhood name is not a quantity to average, and it is not really missing either:
it is written down elsewhere in the same row. Where a column is functionally determined
by another, a gap in it has a correct answer that can be looked up rather than a
plausible answer that has to be invented - and where the determinant is missing too,
the honest outcome is that this file cannot supply the value at all.

Both conclusions are more useful than the median.

The detection is deliberately narrow. Two guards keep it from finding dependencies
everywhere:

* **A near-unique column is not a determinant.** A primary key determines every other
  column by construction; saying so is true and worthless.
* **A handful of rows proves nothing.** Over five records, unrelated columns agree by
  coincidence, so a minimum number of jointly populated rows is required.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from dqcopilot.profiling.type_inference import missing_mask, to_clean_strings


@dataclass(frozen=True, slots=True)
class Dependency:
    """One column whose value is decided by another's."""

    #: The column that decides ("ID_NIL").
    determinant: str
    #: The column being decided ("NIL").
    dependent: str
    #: Rows where both columns are populated, i.e. the evidence for the rule.
    evidence_rows: int
    #: Distinct values of the determinant, i.e. the size of the lookup table.
    distinct_keys: int
    #: Gaps in the dependent column that the determinant can actually fill.
    recoverable_rows: int
    #: Gaps where the determinant is missing too, so nothing here can supply a value.
    unrecoverable_rows: int


#: Below this many jointly populated rows, agreement between two columns is chance.
MIN_EVIDENCE_ROWS = 20

#: A determinant may not have more distinct values than this share of the rows it
#: covers, otherwise it is a key and determines everything trivially.
MAX_KEY_RATIO = 0.5

#: Comparing every column against every other is quadratic. Real files stay well under
#: this; the cap stops a 200-column upload from turning an analysis into a stall.
MAX_COLUMNS_SCANNED = 60


def find_determinant(frame: pd.DataFrame, column: str) -> Dependency | None:
    """Return the column that decides ``column``, or ``None`` when none does.

    When several columns qualify, the one that fills the most gaps wins; ties are broken
    towards the most compact lookup table and then by name, so the result does not depend
    on column order.

    Args:
        frame: The dataset.
        column: The column with gaps to explain.

    Returns:
        The best :class:`Dependency`, or ``None``.
    """
    if column not in frame.columns or frame.shape[1] > MAX_COLUMNS_SCANNED:
        return None

    target = _normalised(frame[column])
    target_gaps = target.isna()
    if not bool(target_gaps.any()) or int(target.nunique()) < 2:
        return None

    candidates: list[Dependency] = []
    for name in frame.columns:
        if name == column:
            continue
        dependency = _test_pair(_normalised(frame[name]), target, name, column)
        if dependency is not None:
            candidates.append(dependency)

    if not candidates:
        return None
    return max(
        candidates,
        key=lambda d: (d.recoverable_rows, -d.distinct_keys, d.determinant),
    )


def build_lookup(frame: pd.DataFrame, dependency: Dependency) -> dict[str, str]:
    """Return the determinant-value to dependent-value mapping.

    Only pairs where both columns are populated contribute, so the table contains no
    invented entries.
    """
    source = _normalised(frame[dependency.determinant])
    target = _normalised(frame[dependency.dependent])
    both = pd.DataFrame({"key": source, "value": target}).dropna()
    pairs = both.drop_duplicates(subset="key")
    return {str(key): str(value) for key, value in zip(pairs["key"], pairs["value"], strict=True)}


def _normalised(series: pd.Series) -> pd.Series:
    """Return the column as stripped strings with every flavour of blank as NA."""
    text = to_clean_strings(series).astype("string").str.strip()
    return text.mask(missing_mask(series) | (text == ""), other=pd.NA)


def _test_pair(
    source: pd.Series,
    target: pd.Series,
    determinant: str,
    dependent: str,
) -> Dependency | None:
    """Return the dependency ``target`` has on ``source``, if it holds."""
    both = pd.DataFrame({"key": source, "value": target}).dropna()
    evidence = int(len(both))
    if evidence < MIN_EVIDENCE_ROWS:
        return None

    distinct_keys = int(both["key"].nunique())
    if distinct_keys < 2 or distinct_keys > MAX_KEY_RATIO * evidence:
        return None

    # The rule itself: every key maps to exactly one value.
    if int(both.groupby("key")["value"].nunique().max()) != 1:
        return None

    # A gap is only recoverable when its key was actually seen alongside a value. A key
    # that appears exclusively on rows where the target is missing teaches us nothing,
    # and counting it would promise a fix that cannot be delivered.
    gaps = target.isna()
    recoverable = int((gaps & source.isin(set(both["key"]))).sum())
    return Dependency(
        determinant=determinant,
        dependent=dependent,
        evidence_rows=evidence,
        distinct_keys=distinct_keys,
        recoverable_rows=recoverable,
        unrecoverable_rows=int(gaps.sum()) - recoverable,
    )
