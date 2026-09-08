# Real public dataset

`milano_attivita_storiche.csv` is a genuine open dataset, committed here unmodified and
used as the project's second demo case. `source.json` records where it came from, when,
and the SHA-256 of the bytes in this directory; `scripts/fetch_public_dataset.py`
refreshes both.

| | |
|---|---|
| Title | Elenco attività storiche e di tradizione nel Comune di Milano |
| Publisher | Regione Lombardia, via the [Comune di Milano open data portal](https://dati.comune.milano.it/dataset/624e5fff-9e40-4adc-9b82-a5d06fa95a26) |
| Licence | [Creative Commons CCZero](http://www.opendefinition.org/licenses/cc-zero) — public domain, redistribution unrestricted |
| Contents | 546 rows, 19 columns: shop name, sign, address, category, year opened |
| Personal data | None. Businesses, addresses and coordinates only |

## Why a second dataset

`data/demo/` is synthetic, and that is a deliberate methodological choice: the generator
knows which rows it broke, so `ground_truth.json` makes the detection rate measurable
instead of anecdotal. It has one unavoidable weakness. The defects it contains are the
defects we thought to inject, which makes any evaluation against it partly circular.

This file closes that gap. Nobody here chose what is wrong with it.

## What it caught

Running the pipeline against it found real defects — a `9999` sentinel in a year column,
a non-Milanese postcode in a file where every row claims Milan, a constant column — and
correctly reported that 17 business rules were skipped rather than passed, because they
were written for the synthetic data and reference columns this file does not have.

It also exposed three things the tool could not do, each of which is now fixed and
covered by `tests/test_encoding_and_placeholders.py`:

1. **Double-encoded text.** Three company names contain UTF-8 read as cp1252. Nothing
   looked wrong: the file parsed, the column profiled as text. The
   `corrupted_encoding` check now proves the corruption by decoding it back.
2. **A placeholder posing as a value.** 28 rows carry `ditta individuale` — a legal
   form — in the company-name field. The cells are populated, so the missing-value
   check never saw them. The `placeholder_values` check now finds repeated filler in
   otherwise-distinct columns, which is what no word list could have anticipated.
3. **Imputation offered on labels.** The tool proposed filling a missing `MUNICIPIO`
   with the median municipality, and a missing coordinate with the median coordinate.
   Both are arithmetically valid and factually wrong about a specific row. Codes,
   identifiers and coordinates now go to manual review instead.

The third one is the useful demo. The bad suggestion was never applied, because a human
has to approve every change — the design absorbed the mistake. A pipeline that corrected
on its own would have written a wrong municipality into the data and reported success.

## Known limits this file still shows

`NIL` (neighbourhood name) is still offered "fill with the most frequent value", which
is wrong for the same reason `MUNICIPIO` was: it is a place, not a quantity. Its name
carries no hint that says so, and the rows where it is missing are exactly the rows
where `ID_NIL` and the coordinates are missing too. Detecting that needs co-missingness
analysis across columns, which this project does not do.
