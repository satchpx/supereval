# Upgrading supereval

## Upgrading the package

```bash
pip install --upgrade supereval
```

No other action is needed for most upgrades.

---

## DuckDB schema migrations

The run history database (`supereval.db`) is migrated automatically on startup. When new columns are added, supereval runs `ALTER TABLE ... ADD COLUMN` in a `try/except` block — if the column already exists, the error is silently ignored.

You don't need to do anything when upgrading. Your existing history is preserved.

**If the database is corrupted or you want a clean slate:**

```bash
rm supereval.db
# The database will be recreated on the next run
```

This deletes all run history. Baselines (`baseline.json` in each dataset directory) are unaffected.

**If you want to keep history but reset the schema:**

This is rarely needed. If you must, export what you need first:

```bash
supereval history list --limit 1000 > history_backup.txt
rm supereval.db
```

---

## Dataset format stability

The dataset format (fields in `cases.jsonl` and `dataset.json`) is defined by the Pydantic models in `supereval/models.py`. Fields are validated on read, so malformed cases will raise `ValidationError` rather than silently corrupt results.

**Adding fields:** New optional fields added to the models have default values. Old `cases.jsonl` files continue to load without changes.

**Removing fields:** If a required field is removed from the schema, old cases that contain it will load fine — Pydantic ignores unknown fields by default.

**Renaming fields:** This is a breaking change. If a required field is renamed, old cases will fail validation. In practice the core field names (`input`, `expected`, `query`, `ground_truth`, `label`, `rubric`, etc.) are stable.

---

## `baseline.json` format stability

`baseline.json` is a plain JSON object written by `supereval run --update-baseline`. The structure is:

```json
{
  "run_id": "run_7d4e9f2a",
  "dataset": "aws-support-qa",
  "ran_at": "2025-11-01T09:14:03+00:00",
  "providers": ["anthropic:claude-opus-4-6"],
  "summary": {
    "total": 5,
    "passed": 5,
    "failed": 0,
    "pass_rate": 1.0,
    "total_cost_usd": 0.0042,
    "avg_latency_ms": 823.0,
    "p50_latency_ms": 791.0,
    "p95_latency_ms": 1241.0
  },
  "cases": [
    {
      "key": "a3f9b2c1d4e5f6a7",
      "vars": {"query": "What is the maximum size of an SQS message?"},
      "passed": true,
      "score": 1.0,
      "latency_ms": 791,
      "cost_usd": 0.0008
    }
  ]
}
```

Cases are matched between runs by a hash of their `vars` dict (`key`). If you change a case's `input` field, its key changes and the baseline match fails — the old case is treated as removed, the new case is treated as new (no regression detected for it on the first run after the change).

This is intentional: if you change the question, you're resetting the baseline for that case.

---

## Promptfoo version

supereval shells out to `promptfoo eval`. Promptfoo's output format has been stable across recent versions, but major version bumps may change the JSON output schema.

If you see parse errors in `_parse_promptfoo_output`, check your Promptfoo version:

```bash
promptfoo --version
```

The output format is parsed from `results.results[].success`, `results.results[].score`, `results.results[].latencyMs`, and `results.stats.tokenUsage`. If these paths change, `supereval/runner.py:_parse_promptfoo_output` needs updating.

---

## Migrating between dataset types

There is no automatic migration between dataset types (e.g. from `qa` to `instruction`). The types have different `input` and `expected` schemas.

If you need to change a dataset's type:

1. Create a new dataset with the correct type
2. Write new cases (the old cases have the wrong schema)
3. Delete the old dataset directory if you no longer need it

---

## Multiple environments (dev vs. prod datasets)

Use `SUPEREVAL_DATASETS_DIR` and `SUPEREVAL_DB_PATH` to keep environments separate:

```bash
# Development
export SUPEREVAL_DATASETS_DIR=./datasets-dev
export SUPEREVAL_DB_PATH=./supereval-dev.db

# Production CI
export SUPEREVAL_DATASETS_DIR=./datasets
export SUPEREVAL_DB_PATH=./supereval.db
```

Both environment variables can also be set in a `.env` file loaded by your CI system.
