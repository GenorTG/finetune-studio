"""CRUD for `benchmark_runs`."""
from __future__ import annotations

import json
import time

from finetune_studio.db.connection import cursor, new_id, row_to_dict


def _get(bid: str) -> dict | None:
    with cursor() as c:
        r = c.execute("SELECT * FROM benchmark_runs WHERE id = ?", (bid,)).fetchone()
    return row_to_dict(r)


def create_benchmark(run_id: str, suite_name: str, scores: dict,
                     time_ms: int = 0) -> dict:
    bid = new_id()
    now = time.time()
    with cursor() as c:
        c.execute(
            "INSERT INTO benchmark_runs (id, run_id, suite_name, scores_json, time_ms, ran_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (bid, run_id, suite_name, json.dumps(scores), time_ms, now),
        )
    return _get(bid)  # type: ignore[return-value]


def get_benchmark(bid: str) -> dict | None:
    return _get(bid)


def list_benchmarks(run_id: str | None = None) -> list[dict]:
    with cursor() as c:
        if run_id:
            rows = c.execute(
                "SELECT * FROM benchmark_runs WHERE run_id = ? ORDER BY ran_at DESC",
                (run_id,),
            ).fetchall()
        else:
            rows = c.execute("SELECT * FROM benchmark_runs ORDER BY ran_at DESC").fetchall()
    return [row_to_dict(r) for r in rows]
