# Data Quality Copilot — one-pager

**A reviewer's tool for messy spreadsheets. It finds the problems and explains them; a
person decides what to fix; everything that happens is recorded.**

---

## The problem

Data arrives from outside your control — a supplier's export, a partner's monthly file,
a form someone filled in. It is almost right. The gap between "almost right" and "usable"
is paid for in analyst hours, and the same file arrives again next month with the same
problems.

The cost is rarely one dramatic failure. It is:

- a revenue figure that is silently wrong because a column was text and half of it did
  not sum;
- a customer counted twice because a name was typed with a trailing space;
- a report nobody trusts, because when a number looks odd nobody can say what was
  changed, by whom, or when.

## Why the obvious tools do not close it

**Manual cleaning in Excel** works and leaves no trace. It is unrepeatable and
unauditable, and the knowledge lives in one person's head.

**Fully automated cleaning tools** solve the wrong half. They are excellent at applying
rules you already know, and dangerous where judgement is needed. A tool that silently
merges `UK` into `United Kingdom` is fine until the day those are genuinely different,
and by then nobody can tell what it did.

The hard part of data quality is not detecting problems. It is deciding what to do about
the ambiguous ones, and being able to explain that decision later.

## What this does

1. **Upload** a CSV or Excel file.
2. **It profiles and checks** — 19 deterministic checks plus business rules you write in
   plain YAML.
3. **It explains** each problem in language a non-engineer can act on: how many rows,
   which rows, and what it breaks downstream.
4. **It proposes** corrections with a before/after preview — and applies **none** of
   them.
5. **You approve** what you want, one by one. Destructive fixes are separated and
   flagged.
6. **You get** a cleaned file, a quality report, and an audit log of every decision —
   including the ones you rejected.

## The design position

| Principle | In practice |
|---|---|
| **Deterministic first** | If a rule can be checked in code, it is. The AI is reserved for genuinely ambiguous questions. |
| **Propose, never apply** | Nothing changes without an explicit approval. Software that guesses on your behalf is the problem, not the solution. |
| **Refuse to guess** | Where a fix would require a coin flip — two equally common spellings, a probable duplicate — it explains the choice and stops. |
| **Record the rejections too** | An audit trail that only records changes cannot answer "was this seen and consciously left alone?" |
| **Your data stays yours** | The dataset is never sent to any API. What goes is column names, types, counts and a few example values per column, with personal-looking columns masked - and only when you ask for it. |

## Where AI is used, and where it is not

Used for four things a rule cannot decide: what an ambiguously-named column holds,
whether two spellings mean the same category, what a finding means for a business
reader, and which rules a dataset ought to have.

Every model response is constrained to a fixed schema, validated, and then **checked
against the actual data**. A suggestion referring to a column or value that does not
exist is discarded, and you can see that it was. AI suggestions are shown separately,
never affect the quality score, and become changes only when you approve them.

The application runs fully without an AI key. Every check, correction, export and audit
record works; only the suggestion panel switches off.

## Where the value shows up

- **Analyst time** moves from hunting for problems to deciding about them.
- **Repeatability** — the same file next month gets the same checks and the same
  documented rules.
- **Traceability** — when a number is questioned, there is an answer.
- **Knowledge capture** — rules that lived in one person's head become a YAML file the
  team owns.

## Honest limitations

This is a demonstration project, and it is more useful to say so.

- Single user, no authentication, no access control.
- In-memory processing, 200,000 rows by default. Not a distributed pipeline.
- The quality score is a transparent, documented heuristic — good for comparing the same
  dataset before and after cleaning, not a validated industry metric and not a KPI.
- Near-duplicate detection matches on a normalised key; it is not fuzzy matching.
- All demonstration data is synthetic. Nothing in it refers to a real person or company.

## Technical summary

Python 3.12 · Streamlit · pandas · Pydantic · SQLAlchemy + Alembic · PostgreSQL ·
Plotly · Anthropic API (optional) · Docker Compose · pytest · Ruff · mypy · GitHub
Actions.

Two containers, `docker compose up --build`, roughly 520 tests including a suite that
verifies every deliberately-seeded error in the synthetic data is actually detected.
