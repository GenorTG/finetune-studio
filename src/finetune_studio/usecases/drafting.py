"""Draft use-case examples with the AI assistant, for review before anything trains on them.

Most people have raw material, not a labelled dataset: tickets without labels, invoices without the extracted fields,
old emails written in the house style. A draft set takes those texts and asks the assistant (the ``drafting`` job,
Settings -> AI assistant) for the missing half of each example:

- ``classification``: one label from the user's list for each text (a reply that is not a label is kept as an empty
  draft with the reason, for the user to pick by hand);
- ``extraction``: one JSON object per text with the user's fields or schema (a reply that is not valid JSON or breaks
  the schema is kept empty with the reason);
- ``style``: each text already sounds right, so it is the answer; the assistant writes the request it answers.

Every draft starts ``pending``. Only rows the user approved (optionally after editing the draft) are turned into
examples, and those go through the normal build (``rows.prepare``): nothing trains on an unreviewed draft, the same
rule as mined pairs. Drafts live next to the use cases (``<project>/usecases/drafts/<id>.json``).
"""
from __future__ import annotations

import json
import os
import re
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from finetune_studio.usecases import schema as schema_mod

DRAFTABLE = ("classification", "extraction", "style")
STATUSES = ("pending", "approved", "rejected")
MAX_TEXTS = 500
MAX_TEXT_CHARS = 6000
_MAX_CONSECUTIVE_ERRORS = 5
_ID = re.compile(r"^[0-9a-f]{8}$")
_LOCK = threading.Lock()

GenerateFn = Callable[[str], str]
ProgressFn = Callable[[dict[str, Any]], None]


class DraftError(ValueError):
    """The draft request cannot run (message shown to the user)."""


@dataclass
class DraftOptions:
    labels: list[str] = field(default_factory=list)
    schema: dict[str, Any] | None = None
    fields: list[str] = field(default_factory=list)
    style_guide: str = ""
    hint: str = ""          # what the texts are (e.g. "support tickets"), passed to the assistant

    def as_dict(self) -> dict[str, Any]:
        return {"labels": self.labels, "schema": self.schema, "fields": self.fields, "style_guide": self.style_guide,
                "hint": self.hint}


# ── input texts ──────────────────────────────────────────────────────────────

_TEXT_KEYS = ("text", "input", "prompt", "answer", "body", "content", "message")


def parse_texts(raw: bytes, filename: str = "") -> list[str]:
    """Texts from an upload: a JSON list (strings or objects with a text field), JSONL, or plain text (blocks separated
    by a blank line; a file without blank lines is one text per line). Empty texts are skipped, duplicates merged."""
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise DraftError("the file is not UTF-8 text") from exc
    items: list[Any]
    stripped = text.strip()
    name = filename.lower()
    if not stripped:
        raise DraftError("the file is empty")
    if name.endswith(".json") or stripped.startswith("["):
        try:
            data = json.loads(stripped)
        except ValueError as exc:
            raise DraftError(f"not valid JSON: {exc}") from exc
        items = (data.get("rows") or data.get("texts") or []) if isinstance(data, dict) else data
        if not isinstance(items, list):
            raise DraftError("a JSON file must be a list of texts (or objects with a text field)")
    elif name.endswith(".jsonl") or stripped.startswith("{"):
        items = []
        for n, line in enumerate(stripped.splitlines(), 1):
            if not line.strip():
                continue
            try:
                items.append(json.loads(line))
            except ValueError as exc:
                raise DraftError(f"line {n} is not valid JSON: {exc}") from exc
    else:
        blocks = re.split(r"\n\s*\n", stripped)
        items = blocks if len(blocks) > 1 else stripped.splitlines()
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        value = item
        if isinstance(item, dict):
            value = next((item[k] for k in _TEXT_KEYS if isinstance(item.get(k), str) and item[k].strip()), "")
        if not isinstance(value, str):
            continue
        value = value.strip()
        key = " ".join(value.lower().split())
        if value and key not in seen:
            seen.add(key)
            out.append(value)
    if not out:
        raise DraftError("no text found (expected strings, or objects with a text / input / prompt field)")
    if len(out) > MAX_TEXTS:
        raise DraftError(f"{len(out)} texts: draft at most {MAX_TEXTS} at a time (split the file)")
    too_long = [i + 1 for i, t in enumerate(out) if len(t) > MAX_TEXT_CHARS]
    if too_long:
        raise DraftError(f"texts longer than {MAX_TEXT_CHARS} characters (numbers {too_long[:10]}): shorten them")
    return out


# ── prompts and reply parsing ────────────────────────────────────────────────

def check_options(template: str, opts: DraftOptions) -> None:
    if template not in DRAFTABLE:
        raise DraftError(f"drafting works for {', '.join(DRAFTABLE)} (distillation has the teacher instead)")
    if template == "classification" and len(opts.labels) < 2:
        raise DraftError("give at least two labels to choose from")
    if template == "extraction":
        if opts.schema is not None:
            schema_mod.check_schema(opts.schema)
        if not opts.fields and not (opts.schema or {}).get("properties"):
            raise DraftError("give the fields to extract (or a JSON schema with properties)")


def _about(opts: DraftOptions) -> str:
    return f"The texts are: {opts.hint}.\n" if opts.hint.strip() else ""


def prompt_for(template: str, text: str, opts: DraftOptions) -> str:
    if template == "classification":
        labels = "\n".join(f"- {label}" for label in opts.labels)
        return (f"{_about(opts)}Classify the text below into exactly one of these labels:\n{labels}\n\n"
                f"Reply with the label only, spelled exactly as listed, and nothing else.\n\nText:\n{text}")
    if template == "extraction":
        if opts.schema is not None:
            spec = "Follow this JSON schema:\n" + json.dumps(opts.schema, ensure_ascii=False)
        else:
            spec = "Use exactly these keys: " + ", ".join(opts.fields) + ". Use null when the text does not say."
        return (f"{_about(opts)}Extract the information from the text below as one JSON object. {spec}\n"
                f"Copy values as written in the text; never invent a value. Reply with the JSON object only: no "
                f"explanation, no code fence.\n\nText:\n{text}")
    guide = f"The writing style: {opts.style_guide}\n" if opts.style_guide.strip() else ""
    return (f"{_about(opts)}{guide}The text below is a reply someone wrote. Write the message or question it answers, "
            f"as the person who asked would have written it: one short request, in the same language as the reply, "
            f"with only the details the reply needs. Reply with that request only.\n\nReply:\n{text}")


def _strip_fence(reply: str) -> str:
    s = reply.strip()
    m = re.match(r"^```(?:json)?\s*(.*?)\s*```$", s, re.DOTALL)
    return m.group(1).strip() if m else s


def parse_reply(template: str, reply: str, opts: DraftOptions) -> tuple[Any, str]:
    """(draft, '') or ('', why the reply was not usable)."""
    s = _strip_fence(reply or "")
    if not s:
        return "", "the assistant returned nothing"
    if template == "classification":
        by_norm = {label.strip().lower(): label for label in opts.labels}
        guess = s.strip().strip(".\"'`* ").splitlines()[0].strip().strip(".\"'`* ")
        if guess.lower() in by_norm:
            return by_norm[guess.lower()], ""
        found = [label for norm, label in by_norm.items() if re.search(rf"(?<!\w){re.escape(norm)}(?!\w)", s.lower())]
        if len(found) == 1:
            return found[0], ""
        return "", f"the reply was not one of the labels: {s[:80]!r}"
    if template == "extraction":
        start, end = s.find("{"), s.rfind("}")
        try:
            value = json.loads(s[start:end + 1]) if start >= 0 and end > start else None
        except ValueError:
            value = None
        if not isinstance(value, dict):
            return "", "the reply was not one JSON object"
        if opts.schema is not None:
            problems = schema_mod.validate(value, opts.schema)
            if problems:
                return value, "does not follow the schema: " + "; ".join(problems[:3])
        return value, ""
    request = s.strip().strip('"').strip()
    if len(request) < 3:
        return "", "the request was too short"
    return request, ""


def draft_rows(template: str, texts: list[str], generate: GenerateFn, opts: DraftOptions, *, workers: int = 1,
               progress: ProgressFn | None = None) -> list[dict[str, Any]]:
    """One assistant call per text; every row comes back ``pending`` (a failed or unusable reply keeps the text with
    an empty draft and the reason, so the user can fill it in)."""
    check_options(template, opts)

    def work(text: str) -> tuple[str, str | None, str]:
        try:
            return text, generate(prompt_for(template, text, opts)), ""
        except Exception as exc:  # noqa: BLE001 - a failed call is reported on its row
            return text, None, f"the assistant call failed: {type(exc).__name__}: {str(exc)[:160]}"

    rows: list[dict[str, Any]] = []
    consecutive = 0
    drafted = 0
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        for n, (text, reply, error) in enumerate(pool.map(work, texts), 1):
            if reply is None:
                consecutive += 1
                if consecutive >= _MAX_CONSECUTIVE_ERRORS and not drafted:
                    raise DraftError(f"the assistant failed {consecutive} calls in a row ({error}); nothing was "
                                     "drafted")
                draft, note = "", error
            else:
                consecutive = 0
                draft, note = parse_reply(template, reply, opts)
                drafted += 1 if draft not in ("", None) and not note else 0
            rows.append({"id": f"r{n:04d}", "input": text, "draft": draft, "note": note, "status": "pending"})
            if progress:
                progress({"done": n, "planned": len(texts), "drafted": drafted})
    return rows


# ── turning approved drafts into examples ────────────────────────────────────

def example_for(template: str, row: dict[str, Any]) -> dict[str, Any]:
    """The upload-shaped example (``rows.prepare`` input) for one approved row."""
    if template == "classification":
        return {"input": row["input"], "label": row["draft"]}
    if template == "extraction":
        return {"input": row["input"], "output": row["draft"]}
    return {"prompt": row["draft"], "answer": row["input"]}


def approved_examples(draft_set: dict[str, Any]) -> list[dict[str, Any]]:
    return [example_for(draft_set["template"], r) for r in draft_set["rows"] if r["status"] == "approved"]


def check_row_edit(template: str, opts: DraftOptions, value: Any) -> tuple[Any, str]:
    """A user's edited draft, normalised, or ('', why it is not usable)."""
    if template == "classification":
        value = str(value or "").strip()
        match = next((label for label in opts.labels if label.lower() == value.lower()), None)
        return (match, "") if match else ("", f"{value!r} is not one of the labels")
    if template == "extraction":
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except ValueError:
                return "", "not valid JSON"
        if not isinstance(value, dict):
            return "", "must be a JSON object"
        if opts.schema is not None:
            problems = schema_mod.validate(value, opts.schema)
            if problems:
                return "", "does not follow the schema: " + "; ".join(problems[:3])
        return value, ""
    value = str(value or "").strip()
    return (value, "") if len(value) >= 3 else ("", "the request is too short")


# ── storage ──────────────────────────────────────────────────────────────────

def drafts_dir(pid: str) -> Path:
    from finetune_studio.usecases.store import usecases_dir

    return usecases_dir(pid) / "drafts"


def _path(pid: str, did: str) -> Path:
    if not _ID.match(did or ""):
        raise LookupError("draft set not found")
    return drafts_dir(pid) / f"{did}.json"


def _write(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, path)


def counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    out = {s: 0 for s in STATUSES}
    for r in rows:
        out[r["status"]] = out.get(r["status"], 0) + 1
    out["empty"] = sum(1 for r in rows if r["draft"] in ("", None))
    return out


def create(pid: str, name: str, template: str, opts: DraftOptions, rows: list[dict[str, Any]],
           assistant: dict[str, Any]) -> dict[str, Any]:
    from finetune_studio.db.connection import new_id

    did = new_id()[:8]
    data = {"id": did, "name": name or f"{template} drafts", "template": template, "options": opts.as_dict(),
            "assistant": assistant, "created_at": time.time(), "rows": rows}
    with _LOCK:
        _write(_path(pid, did), data)
    return data


def get(pid: str, did: str) -> dict[str, Any]:
    path = _path(pid, did)
    if not path.is_file():
        raise LookupError("draft set not found")
    return json.loads(path.read_text(encoding="utf-8"))


def options_of(draft_set: dict[str, Any]) -> DraftOptions:
    o = draft_set.get("options") or {}
    return DraftOptions(labels=list(o.get("labels") or []), schema=o.get("schema"), fields=list(o.get("fields") or []),
                        style_guide=str(o.get("style_guide") or ""), hint=str(o.get("hint") or ""))


def summary(draft_set: dict[str, Any]) -> dict[str, Any]:
    return {k: draft_set.get(k) for k in ("id", "name", "template", "created_at", "assistant")} | {
        "counts": counts(draft_set["rows"]), "total": len(draft_set["rows"])}


def list_all(pid: str) -> list[dict[str, Any]]:
    d = drafts_dir(pid)
    if not d.is_dir():
        return []
    out = []
    for p in sorted(d.glob("*.json")):
        try:
            out.append(summary(json.loads(p.read_text(encoding="utf-8"))))
        except (OSError, ValueError, KeyError):
            continue
    return sorted(out, key=lambda s: s.get("created_at") or 0, reverse=True)


def update_rows(pid: str, did: str, changes: list[dict[str, Any]]) -> dict[str, Any]:
    """Apply ``[{id, status?, draft?}]``. An edited draft is checked first; approving a row with no usable draft is
    refused (``DraftError`` naming the row)."""
    with _LOCK:
        data = get(pid, did)
        opts = options_of(data)
        by_id = {r["id"]: r for r in data["rows"]}
        for change in changes:
            row = by_id.get(str(change.get("id") or ""))
            if row is None:
                raise LookupError(f"row {change.get('id')!r} not found")
            if "draft" in change:
                value, why = check_row_edit(data["template"], opts, change["draft"])
                if why:
                    raise DraftError(f"{row['id']}: {why}")
                row["draft"], row["note"] = value, ""
            status = change.get("status")
            if status is not None:
                if status not in STATUSES:
                    raise DraftError(f"status must be one of {', '.join(STATUSES)}")
                if status == "approved" and (row["draft"] in ("", None) or row.get("note")):
                    raise DraftError(f"{row['id']}: fill in a usable draft before approving it")
                row["status"] = status
        _write(_path(pid, did), data)
        return data


def delete(pid: str, did: str) -> None:
    path = _path(pid, did)
    if not path.is_file():
        raise LookupError("draft set not found")
    path.unlink()
