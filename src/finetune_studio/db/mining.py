"""Durable record of per-file mining: jobs, the file's data-point inventory, and every model round.

A mining job works on ONE parsed file (``source_id``) and moves through fixed stages
(``data.mining.job``). Everything it decides is stored here, so a run can be diagnosed later:

- ``mining_jobs``   one row per job (kind ``mine`` builds the inventory and mines; ``fill`` closes the open items of a
                    reviewed file), with its stage, counters and the helper that ran it.
- ``mining_items``  the inventory: every data point the file holds (rule-based statements + agent-listed facts). An item
                    is covered when a pair states its values; a reviewer can exclude an item with a reason.
- ``mining_rounds`` every model call: what was asked (messages), the raw reply, timing and how many pairs/items it gave.
"""
from __future__ import annotations

import json
import time
from collections.abc import Iterable
from typing import Any

from finetune_studio.db.connection import cursor, new_id, row_to_dict

ACTIVE_STATUSES = ("queued", "running")
_JOB_FIELDS = {"status", "stage", "started_at", "finished_at", "error", "stats_json", "helper_json", "message", "runner",
               "cancel_requested_at"}


# ── jobs ────────────────────────────────────────────────────────────────────────────────────────────────────────────

def create_job(project_id: str, source_id: str, filename: str, kind: str, helper: dict[str, Any] | None = None) -> dict:
    jid = new_id()
    with cursor() as c:
        c.execute(
            "INSERT INTO mining_jobs (id, project_id, source_id, filename, kind, status, stage, helper_json, created_at) "
            "VALUES (?, ?, ?, ?, ?, 'queued', 'queued', ?, ?)",
            (jid, project_id, source_id, filename, kind, json.dumps(helper or {}), time.time()),
        )
    return get_job(jid)  # type: ignore[return-value]


def get_job(job_id: str) -> dict | None:
    with cursor() as c:
        return row_to_dict(c.execute("SELECT * FROM mining_jobs WHERE id = ?", (job_id,)).fetchone())


def update_job(job_id: str, **fields: Any) -> dict | None:
    sets, vals = [], []
    for key, value in fields.items():
        if key not in _JOB_FIELDS:
            raise KeyError(f"mining_jobs has no writable field {key!r}")
        if key.endswith("_json") and not isinstance(value, str):
            value = json.dumps(value)
        sets.append(f"{key} = ?")
        vals.append(value)
    if sets:
        with cursor() as c:
            c.execute(f"UPDATE mining_jobs SET {', '.join(sets)} WHERE id = ?", [*vals, job_id])
    return get_job(job_id)


def list_jobs(project_id: str | None = None, *, source_id: str | None = None, active: bool = False,
              limit: int = 200) -> list[dict]:
    where, args = [], []
    if project_id:
        where.append("project_id = ?")
        args.append(project_id)
    if source_id:
        where.append("source_id = ?")
        args.append(source_id)
    if active:
        where.append(f"status IN ({', '.join('?' for _ in ACTIVE_STATUSES)})")
        args.extend(ACTIVE_STATUSES)
    sql = "SELECT * FROM mining_jobs" + (f" WHERE {' AND '.join(where)}" if where else "")
    with cursor() as c:
        rows = c.execute(sql + " ORDER BY created_at DESC LIMIT ?", [*args, limit]).fetchall()
    return [row_to_dict(r) for r in rows]  # type: ignore[misc]


def interrupt_active_jobs(reason: str) -> int:
    """After a process restart nothing is running any more: say so on every job that claimed to be."""
    now = time.time()
    with cursor() as c:
        n = c.execute(
            f"UPDATE mining_jobs SET status = 'interrupted', error = ?, finished_at = ? "
            f"WHERE status IN ({', '.join('?' for _ in ACTIVE_STATUSES)})",
            (reason, now, *ACTIVE_STATUSES),
        ).rowcount
    return int(n or 0)


def claim_queued(runner: str, limit: int) -> list[dict]:
    """Oldest queued jobs, each taken by ``runner`` (a mining worker job id) in one conditional UPDATE: a job is never
    taken twice, even by two workers at once. Claimed jobs are ``running`` from here on."""
    if limit <= 0:
        return []
    with cursor() as c:
        ids = [r["id"] for r in c.execute(
            "SELECT id FROM mining_jobs WHERE status = 'queued' ORDER BY created_at LIMIT ?", (limit,)).fetchall()]
    claimed = []
    now = time.time()
    for jid in ids:
        with cursor() as c:
            n = c.execute("UPDATE mining_jobs SET status = 'running', runner = ?, started_at = ? "
                          "WHERE id = ? AND status = 'queued'", (runner, now, jid)).rowcount
        if n:
            claimed.append(get_job(jid))
    return [j for j in claimed if j]


def cancel_queued(job_id: str) -> bool:
    """Cancel a job no worker has taken yet (one statement). False when it is not queued any more."""
    with cursor() as c:
        n = c.execute("UPDATE mining_jobs SET status = 'cancelled', stage = 'done', message = 'Cancelled.', "
                      "finished_at = ? WHERE id = ? AND status = 'queued'", (time.time(), job_id)).rowcount
    return bool(n)


def request_cancel(job_id: str) -> bool:
    """Ask the worker running this job to stop it. False when it is not running any more."""
    with cursor() as c:
        n = c.execute("UPDATE mining_jobs SET cancel_requested_at = ?, message = 'Cancelling…' "
                      "WHERE id = ? AND status = 'running'", (time.time(), job_id)).rowcount
    return bool(n)


def cancel_requested(job_ids: Iterable[str]) -> set[str]:
    ids = list(job_ids)
    if not ids:
        return set()
    with cursor() as c:
        rows = c.execute(f"SELECT id FROM mining_jobs WHERE cancel_requested_at IS NOT NULL "
                         f"AND id IN ({', '.join('?' for _ in ids)})", ids).fetchall()
    return {r["id"] for r in rows}


def runners_of_running() -> set[str]:
    """Worker job ids that hold a running job ('' for jobs from before mining ran as a job)."""
    with cursor() as c:
        return {r["runner"] for r in c.execute("SELECT DISTINCT runner FROM mining_jobs WHERE status = 'running'")}


def interrupt_running(reason: str, *, keep_runners: Iterable[str] = (), runner: str | None = None) -> int:
    """Running jobs whose worker is gone become ``interrupted`` (items and pairs stay; Fill gaps continues them).
    ``keep_runners``: workers that are alive. ``runner``: only that worker's jobs. Queued jobs are never touched."""
    keep = list(keep_runners)
    where, args = ["status = 'running'"], [reason, time.time()]
    if runner is not None:
        where.append("runner = ?")
        args.append(runner)
    if keep:
        where.append(f"runner NOT IN ({', '.join('?' for _ in keep)})")
        args.extend(keep)
    with cursor() as c:
        n = c.execute(f"UPDATE mining_jobs SET status = 'interrupted', error = ?, finished_at = ? "
                      f"WHERE {' AND '.join(where)}", args).rowcount
    return int(n or 0)


# ── inventory items ─────────────────────────────────────────────────────────────────────────────────────────────────

def insert_items(project_id: str, source_id: str, job_id: str, items: Iterable[dict[str, Any]]) -> list[str]:
    now = time.time()
    ids: list[str] = []
    with cursor() as c:
        for it in items:
            iid = new_id() + new_id()
            c.execute(
                "INSERT INTO mining_items (id, project_id, source_id, job_id, chunk_idx, origin, kind, text, header, "
                "values_json, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (iid, project_id, source_id, job_id, int(it["chunk_idx"]), it["origin"], it["kind"], it["text"],
                 it.get("header", ""), json.dumps(list(it.get("values") or [])), now),
            )
            ids.append(iid)
    return ids


def list_items(project_id: str, source_id: str | None = None) -> list[dict]:
    sql, args = "SELECT * FROM mining_items WHERE project_id = ?", [project_id]
    if source_id:
        sql += " AND source_id = ?"
        args.append(source_id)
    with cursor() as c:
        rows = c.execute(sql + " ORDER BY source_id, chunk_idx, created_at, rowid", args).fetchall()
    return [row_to_dict(r) for r in rows]  # type: ignore[misc]


def get_item(project_id: str, item_id: str) -> dict | None:
    with cursor() as c:
        return row_to_dict(c.execute("SELECT * FROM mining_items WHERE project_id = ? AND id = ?",
                                     (project_id, item_id)).fetchone())


def set_item_exclusion(project_id: str, item_id: str, reason: str) -> dict | None:
    """``reason`` non-empty: the reviewer decided this is not a trainable fact; empty: include it again (which also
    drops a helper proposal to exclude it: the reviewer disagreed)."""
    with cursor() as c:
        if reason:
            c.execute("UPDATE mining_items SET excluded_reason = ?, excluded_at = ? WHERE project_id = ? AND id = ?",
                      (reason, time.time(), project_id, item_id))
        else:
            c.execute("UPDATE mining_items SET excluded_reason = '', excluded_at = NULL, proposed_reason = '' "
                      "WHERE project_id = ? AND id = ?", (project_id, item_id))
    return get_item(project_id, item_id)


def propose_exclusions(project_id: str, reasons: dict[str, str]) -> int:
    """The helper's verdict "not a trainable fact" per item id. Never excludes: a reviewer confirms (``confirm_proposals``)."""
    with cursor() as c:
        for iid, reason in reasons.items():
            c.execute("UPDATE mining_items SET proposed_reason = ? WHERE project_id = ? AND id = ? AND excluded_reason = ''",
                      (reason[:300], project_id, iid))
    return len(reasons)


def confirm_proposals(project_id: str, source_id: str, item_ids: Iterable[str] | None = None) -> int:
    """Turn the helper's proposals of a file (or of ``item_ids``) into exclusions; returns how many."""
    sql = ("UPDATE mining_items SET excluded_reason = 'helper: ' || proposed_reason, excluded_at = ? "
           "WHERE project_id = ? AND source_id = ? AND proposed_reason != '' AND excluded_reason = ''")
    args: list[Any] = [time.time(), project_id, source_id]
    if item_ids is not None:
        ids = list(item_ids)
        if not ids:
            return 0
        sql += f" AND id IN ({','.join('?' * len(ids))})"
        args += ids
    with cursor() as c:
        return c.execute(sql, args).rowcount


def delete_file_records(project_id: str, source_id: str) -> None:
    """Forget a file's inventory and rounds (a fresh mine rebuilds them). Jobs stay as history."""
    with cursor() as c:
        c.execute("DELETE FROM mining_items WHERE project_id = ? AND source_id = ?", (project_id, source_id))
        c.execute("DELETE FROM mining_rounds WHERE project_id = ? AND source_id = ?", (project_id, source_id))


# ── rounds ──────────────────────────────────────────────────────────────────────────────────────────────────────────

def add_round(*, project_id: str, source_id: str, job_id: str, chunk_idx: int, kind: str, model: str,
              messages: list[dict[str, str]], item_ids: list[str] | None = None) -> str:
    rid = new_id() + new_id()
    with cursor() as c:
        c.execute(
            "INSERT INTO mining_rounds (id, project_id, source_id, job_id, chunk_idx, kind, model, messages_json, "
            "item_ids_json, started_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (rid, project_id, source_id, job_id, chunk_idx, kind, model, json.dumps(messages),
             json.dumps(item_ids or []), time.time()),
        )
    return rid


def finish_round(round_id: str, *, reply: str = "", error: str = "", seconds: float = 0.0, parsed: int = 0,
                 accepted: int = 0, items_added: int = 0) -> None:
    with cursor() as c:
        c.execute(
            "UPDATE mining_rounds SET reply = ?, error = ?, seconds = ?, parsed = ?, accepted = ?, items_added = ? "
            "WHERE id = ?",
            (reply, error[:2000], round(seconds, 2), parsed, accepted, items_added, round_id),
        )


_ROUND_SUMMARY = ("id, project_id, source_id, job_id, chunk_idx, kind, model, error, started_at, seconds, parsed, "
                  "accepted, items_added")


def list_rounds(project_id: str, source_id: str, job_id: str | None = None) -> list[dict]:
    """Rounds without their (large) prompts and replies, oldest first."""
    sql = f"SELECT {_ROUND_SUMMARY} FROM mining_rounds WHERE project_id = ? AND source_id = ?"
    args: list[Any] = [project_id, source_id]
    if job_id:
        sql += " AND job_id = ?"
        args.append(job_id)
    with cursor() as c:
        rows = c.execute(sql + " ORDER BY started_at, rowid", args).fetchall()
    return [row_to_dict(r) for r in rows]  # type: ignore[misc]


def get_round(project_id: str, round_id: str) -> dict | None:
    with cursor() as c:
        return row_to_dict(c.execute("SELECT * FROM mining_rounds WHERE project_id = ? AND id = ?",
                                     (project_id, round_id)).fetchone())
