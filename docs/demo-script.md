# 90-second demo script

For a booth conversation at Big Data & AI Paris. The aim is not to show every feature —
it is to land one idea: **the tool finds the problems and explains them, but the human
decides, and every decision is recorded.**

## Before you start

```bash
docker compose up --build          # or: streamlit run app/streamlit_app.py
```

- Browser open at <http://localhost:8501>, window already sized.
- `data/demo/customers.csv` ready to drag in.
- If you want the AI panel: `ANTHROPIC_API_KEY` set **before** starting. If not, say so —
  reduced mode is part of the story, not an apology.
- Zoom to ~110% so the severity colours read from a metre away.

---

## The script

### 0:00 — 0:15 · The problem

> "Every analytics team has the same afternoon. A file arrives, someone opens it in
> Excel, and spends two hours finding that `country` has both `FR` and `France`, revenue
> is text so it won't sum, and two rows are the same supplier typed twice. They fix some
> of it, keep no record, and next month the same file arrives."

**Do:** drag `customers.csv` onto the uploader.

### 0:15 — 0:35 · What it found

**Do:** land on the Dashboard. Point at the gauge, then the per-column bars.

> "220 rows, nine columns, analysed in under a second. Score 67 out of 100 — and that
> formula is printed right here, it's not a black box. The worst column is `signup_date`
> at 38, because it mixes ISO dates with `DD/MM/YYYY` and contains a 30th of February."

**Do:** open the Findings tab, expand one finding.

> "Every finding says how many rows, which rows, and why it matters. Not '3 values
> invalid' — 'mixed formats are dangerous because 01/02/2024 means two different days
> depending on the convention'."

### 0:35 — 0:60 · The part that matters

**Do:** open the Corrections tab. Expand a whitespace fix and show the before/after
preview. Scroll to "Needs a closer look".

> "Here's the part I care about. Nothing has been changed. Every correction is a
> proposal with a before-and-after preview, and every checkbox starts unticked.
>
> Notice the split: safe and reversible at the top, destructive and AI-sourced below.
> And down here — probable duplicates — the tool refuses to guess. It found them, it
> explains them, and it hands you the decision, because merging customer records is not
> something software should do on its own."

**Do:** tick two corrections, press Apply.

> "Score 67 to 71. That's measured, not predicted — the cleaned dataset goes through the
> whole pipeline again."

### 1:00 — 1:15 · The AI, kept in its lane

**Do:** open the AI suggestions tab. Expand "Show exactly what would be sent".

> "The AI does the ambiguous work only: does `France` mean `FR` in this dataset? And the
> dataset never leaves the machine — this is the entire payload, and columns that look
> personal are masked to their shape.
>
> Every response is schema-constrained, Pydantic-validated, and then checked against
> your actual data. If it proposes merging a value that isn't in the column, it's
> dropped, and you can see why. AI findings never move the score."

*(No API key? Show the reduced-mode panel instead: "no key, and everything except this
panel still works. That's deliberate.")*

### 1:15 — 1:30 · The trail

**Do:** open Downloads. Click the report.

> "Cleaned file, a self-contained HTML report, and a JSON audit log recording every
> correction offered, every decision taken — including the ones you rejected — and every
> change applied. That's stored in PostgreSQL too, as metadata only: it can tell you
> five emails were malformed and which rows, not what they said.
>
> Six days, about 380 tests, and the one I'd point you at asserts that every deliberately
> seeded error in the demo data actually gets found."

---

## If they ask

**"Why not just use Great Expectations?"**
> Different job. GE validates a dataset you already understand against expectations you
> already wrote. This is for the first hour, when you don't know what's wrong yet. The
> natural next step is exporting approved rules *as* a GE suite.

**"Does the AI change my data?"**
> No, and it structurally can't. An AI suggestion becomes an ordinary proposal that you
> tick, and the only module allowed to change a DataFrame acts solely on approved ones.

**"How big a file?"**
> 200,000 rows by default, in memory. It's not a distributed pipeline and doesn't claim
> to be.

**"Is the score standard?"**
> No, and I'd rather say that clearly. It's a documented heuristic, useful for the same
> dataset before and after, or for ranking columns within one dataset. Not a KPI.

**"What would you do next?"**
> Rule editing in the UI, then quality trend over time — the schema already stores
> per-column scores per analysis, so that's a query and a chart, not new plumbing.

---

## Timing notes

- The whole thing fits in 90 seconds only if you do **not** narrate the profiling tab.
  Skip it; it's the least surprising screen.
- If you have three minutes, add: open `config/business_rules.yaml` and show that a
  domain expert can add a rule without touching Python.
- Never apologise for reduced mode. "It works without an API key" is a feature.
