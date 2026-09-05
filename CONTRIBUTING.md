# Contributing

Thanks for taking a look. This is a portfolio project, so the bar for a change is
simply: does it make the project clearer, more correct, or more honest about its limits?

## Getting set up

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
cp .env.example .env
python scripts/generate_demo_data.py
streamlit run app/streamlit_app.py
```

Python 3.12 or newer. No database is needed for development: the default
`DATABASE_URL` is a local SQLite file, and the tables are created on first run.

## Before opening a pull request

All four must pass. CI runs exactly these:

```bash
ruff format .
ruff check .
mypy
pytest
```

## House rules

These are the conventions that keep the project coherent. They are worth reading before
adding a feature.

**Deterministic first.** If a rule can be checked with Python, it is checked with
Python. The language model is reserved for genuinely ambiguous questions - what a column
means, whether two spellings denote the same category. A new check that could have been
a regex should be a regex.

**Nothing changes data without approval.** Corrections are *proposed*, never applied.
`dqcopilot.corrections.applier` is the only module allowed to mutate a DataFrame, and it
only acts on proposals with an explicit `APPROVED` decision. If a fix would require
guessing, propose `MANUAL_REVIEW` and explain the choice instead of picking one.

**AI output is untrusted input.** Every response is constrained to a Pydantic schema
*and* checked against the real dataset in `dqcopilot.ai.grounding`. A suggestion that
names a column or a value that does not exist is dropped, with a reason the user can
read. AI findings never affect the quality score.

**The dataset does not leave the machine.** Only the shape description built by
`dqcopilot.ai.payload` is sent to Anthropic, and columns that look personal have their
example values masked. If you add a new AI task, add its payload builder there and say
in the docstring exactly what it sends.

**Log metadata, never values.** Counts, ratios, row numbers and durations are fine. Cell
values are not. The same rule governs the database: `redact_details` strips examples
before anything is stored.

## Adding a new check

1. Put it in `src/dqcopilot/validation/checks/`, subclassing `Check` and decorated with
   `@register_check`.
2. Give it a stable `check_id`; it becomes part of the finding id, which is what lets a
   user's approval survive a re-run.
3. Write an `explanation` a non-engineer can act on. "3 values are invalid" is not
   enough - say which rows, why it matters, and what it breaks downstream.
4. Add a proposal builder in `corrections/proposer.py` if the fix is safe to automate,
   or let it fall through to `MANUAL_REVIEW` if it is not.
5. Seed the defect in `demodata/generator.py`, record it via `log.record(...)`, and add
   its issue type to `ACCEPTABLE_DETECTORS` in `tests/test_demo_data_detection.py`.
   That test asserts every seeded defect is found, and it is what stops the check suite
   from silently degrading.
6. Add unit tests covering the positive case, the negative case, and at least one edge
   case (empty column, single row, all-null).

## Testing conventions

- No test may make a network call. The Anthropic SDK is replaced by a stub in
  `tests/test_ai.py`; follow that pattern for anything new.
- Database tests run against a temporary SQLite file, never a shared server.
- Prefer a test that states a guarantee ("rejections are kept, not discarded") over one
  that restates the implementation.

## Changing the database schema

```bash
alembic revision --autogenerate -m "what changed"
alembic upgrade head
```

Review the generated file before committing - autogenerate is a starting point, not an
answer. CI fails if the models and the migrations disagree.

## What this project deliberately does not have

Kubernetes, Redis, Celery, authentication, multi-tenancy, and microservices. Each was
considered and left out: none of them earns its complexity for a single-user demo, and
every one would be another thing to keep working. If you want to add one, the pull
request should explain what stopped working without it.
