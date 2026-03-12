"""
Persistent run history store backed by DuckDB.

The database file defaults to ./supereval.db (relative to cwd).
Override with SUPEREVAL_DB_PATH env var.

Schema
------
runs         — one row per eval run (summary-level metrics)
case_results — one row per case per run (drill-down)
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from .runner import RunResult


def db_path() -> Path:
    env = os.environ.get("SUPEREVAL_DB_PATH")
    return Path(env) if env else Path.cwd() / "supereval.db"


def _connect():
    import duckdb
    conn = duckdb.connect(str(db_path()))
    _ensure_schema(conn)
    return conn


def _ensure_schema(conn) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS runs (
            run_id            TEXT PRIMARY KEY,
            dataset           TEXT    NOT NULL,
            ran_at            TEXT    NOT NULL,
            providers         TEXT    NOT NULL,   -- JSON array (LLM) or runner_id (agent)
            total_cases       INTEGER NOT NULL,
            passed            INTEGER NOT NULL,
            failed            INTEGER NOT NULL,
            pass_rate         DOUBLE  NOT NULL,
            total_cost_usd    DOUBLE  NOT NULL DEFAULT 0,
            total_tokens      INTEGER NOT NULL DEFAULT 0,
            prompt_tokens     INTEGER NOT NULL DEFAULT 0,
            completion_tokens INTEGER NOT NULL DEFAULT 0,
            avg_latency_ms    DOUBLE  NOT NULL DEFAULT 0,
            p50_latency_ms    DOUBLE  NOT NULL DEFAULT 0,
            p95_latency_ms    DOUBLE  NOT NULL DEFAULT 0,
            run_type          TEXT    NOT NULL DEFAULT 'llm'
        )
    """)
    # Migrate existing databases that pre-date the run_type column
    try:
        conn.execute("ALTER TABLE runs ADD COLUMN run_type TEXT DEFAULT 'llm'")
    except Exception:
        pass  # column already exists

    conn.execute("CREATE SEQUENCE IF NOT EXISTS case_results_id_seq START 1")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS case_results (
            id         BIGINT DEFAULT nextval('case_results_id_seq') PRIMARY KEY,
            run_id     TEXT    NOT NULL,
            dataset    TEXT    NOT NULL,
            vars_key   TEXT    NOT NULL,
            vars_json  TEXT    NOT NULL,
            passed     BOOLEAN NOT NULL,
            score      DOUBLE  NOT NULL,
            latency_ms INTEGER NOT NULL DEFAULT 0,
            cost_usd   DOUBLE  NOT NULL DEFAULT 0
        )
    """)


# ---------------------------------------------------------------------------
# Write
# ---------------------------------------------------------------------------

def record_run(result: RunResult) -> None:
    """Persist a RunResult to the history store. Idempotent on run_id."""
    conn = _connect()
    try:
        conn.execute(
            """
            INSERT OR REPLACE INTO runs VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
            """,
            [
                result.run_id,
                result.dataset,
                result.ran_at,
                json.dumps(result.providers),
                result.total,
                result.passed,
                result.failed,
                round(result.pass_rate, 6),
                round(result.total_cost_usd, 6),
                result.total_tokens,
                result.prompt_tokens,
                result.completion_tokens,
                round(result.avg_latency_ms, 2),
                round(result.p50_latency_ms, 2),
                round(result.p95_latency_ms, 2),
                "llm",
            ],
        )
        for case in result.cases:
            conn.execute(
                """
                INSERT INTO case_results
                    (run_id, dataset, vars_key, vars_json, passed, score, latency_ms, cost_usd)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    result.run_id,
                    result.dataset,
                    case.key,
                    json.dumps(case.vars, ensure_ascii=False),
                    case.passed,
                    round(case.score, 6),
                    case.latency_ms,
                    round(case.cost_usd, 6),
                ],
            )
    finally:
        conn.close()


def record_agent_run(result) -> None:
    """Persist an AgentRunResult to the history store. Idempotent on run_id."""
    conn = _connect()
    try:
        conn.execute(
            """
            INSERT OR REPLACE INTO runs VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
            """,
            [
                result.run_id,
                result.dataset,
                result.ran_at,
                json.dumps([result.runner_id]) if result.runner_id else "[]",
                result.total,
                result.passed,
                result.failed,
                round(result.pass_rate, 6),
                round(result.total_cost_usd, 6),
                0,   # total_tokens — not tracked at agent level
                0,   # prompt_tokens
                0,   # completion_tokens
                round(result.avg_latency_ms, 2),
                round(result.p50_latency_ms, 2),
                round(result.p95_latency_ms, 2),
                "agent",
            ],
        )
        for case in result.cases:
            conn.execute(
                """
                INSERT INTO case_results
                    (run_id, dataset, vars_key, vars_json, passed, score, latency_ms, cost_usd)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    result.run_id,
                    result.dataset,
                    case.case_id,
                    json.dumps(case.vars, ensure_ascii=False),
                    case.passed,
                    round(case.score.composite_score, 6),
                    case.latency_ms,
                    round(case.cost_usd, 6),
                ],
            )
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Read
# ---------------------------------------------------------------------------

def list_runs(
    dataset: str | None = None,
    limit: int = 20,
    run_type: str | None = None,
) -> list[dict]:
    """Return recent runs, optionally filtered by dataset and/or run_type."""
    conn = _connect()
    try:
        conditions = []
        params: list = []
        if dataset:
            conditions.append("dataset = ?")
            params.append(dataset)
        if run_type:
            conditions.append("run_type = ?")
            params.append(run_type)

        where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
        params.append(limit)

        rows = conn.execute(
            f"""
            SELECT run_id, dataset, ran_at, providers, total_cases, passed, failed,
                   pass_rate, total_cost_usd, total_tokens, avg_latency_ms, p95_latency_ms,
                   run_type
            FROM runs
            {where}
            ORDER BY ran_at DESC
            LIMIT ?
            """,
            params,
        ).fetchall()
        cols = [
            "run_id", "dataset", "ran_at", "providers", "total_cases",
            "passed", "failed", "pass_rate", "total_cost_usd", "total_tokens",
            "avg_latency_ms", "p95_latency_ms", "run_type",
        ]
        return [dict(zip(cols, row)) for row in rows]
    finally:
        conn.close()


def get_run(run_id: str) -> dict | None:
    """Return full run record or None if not found."""
    conn = _connect()
    try:
        row = conn.execute(
            "SELECT * FROM runs WHERE run_id = ?", [run_id]
        ).fetchone()
        if row is None:
            return None
        cols = [d[0] for d in conn.description]
        return dict(zip(cols, row))
    finally:
        conn.close()


def get_run_cases(run_id: str) -> list[dict]:
    """Return all case results for a run."""
    conn = _connect()
    try:
        rows = conn.execute(
            """
            SELECT vars_json, passed, score, latency_ms, cost_usd
            FROM case_results
            WHERE run_id = ?
            ORDER BY id
            """,
            [run_id],
        ).fetchall()
        return [
            {
                "vars": json.loads(r[0]),
                "passed": r[1],
                "score": r[2],
                "latency_ms": r[3],
                "cost_usd": r[4],
            }
            for r in rows
        ]
    finally:
        conn.close()


def get_stats(dataset: str, last: int = 10) -> dict:
    """
    Return aggregate cost/latency/pass-rate stats over the last N runs
    for a dataset.
    """
    conn = _connect()
    try:
        row = conn.execute(
            """
            WITH recent AS (
                SELECT pass_rate, total_cost_usd, avg_latency_ms, p95_latency_ms, total_tokens
                FROM runs
                WHERE dataset = ?
                ORDER BY ran_at DESC
                LIMIT ?
            )
            SELECT
                COUNT(*)                        AS run_count,
                AVG(pass_rate)                  AS avg_pass_rate,
                MIN(pass_rate)                  AS min_pass_rate,
                MAX(pass_rate)                  AS max_pass_rate,
                SUM(total_cost_usd)             AS total_cost_usd,
                AVG(total_cost_usd)             AS avg_cost_per_run,
                MAX(total_cost_usd)             AS max_cost_per_run,
                AVG(avg_latency_ms)             AS avg_latency_ms,
                MAX(p95_latency_ms)             AS max_p95_latency_ms,
                SUM(total_tokens)               AS total_tokens
            FROM recent
            """,
            [dataset, last],
        ).fetchone()

        cols = [
            "run_count", "avg_pass_rate", "min_pass_rate", "max_pass_rate",
            "total_cost_usd", "avg_cost_per_run", "max_cost_per_run",
            "avg_latency_ms", "max_p95_latency_ms", "total_tokens",
        ]
        return dict(zip(cols, row)) if row else {}
    finally:
        conn.close()
