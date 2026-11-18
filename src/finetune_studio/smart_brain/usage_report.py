"""Untrusted diagnostic report -> bounded, inert Studio document suggestions."""
from __future__ import annotations

import json
import math
import os
import tempfile
import time
from pathlib import Path

MAX_REPORT_BYTES = 32 * 1024 * 1024


def suggestions(raw: bytes) -> dict:
    if len(raw) > MAX_REPORT_BYTES:
        raise ValueError("report exceeds 32 MiB")
    try:
        body = json.loads(raw)
    except (ValueError, UnicodeError, RecursionError) as error:
        raise ValueError("upload a UTF-8 JSON missing-knowledge report") from error
    if not isinstance(body, dict) or body.get("schema") != "smart-brain-usage-v1":
        raise ValueError("unsupported report schema")
    gaps = body.get("gaps")
    if not isinstance(gaps, list) or len(gaps) > 10000:
        raise ValueError("gaps must be a list of at most 10000 events")
    grouped = {}
    for row in gaps:
        if not isinstance(row, dict):
            raise ValueError("each gap must be an event object")  # noqa: TRY004 — malformed external JSON
        query = row.get("query")
        if query is not None and (not isinstance(query, str) or len(query) > 2000):
            raise ValueError("gap query must be null or text up to 2000 characters")
        for key in ("not_found", "weak"):
            if type(row.get(key)) not in (bool, int) or row[key] not in (0, 1):
                raise ValueError("gap flags must be 0 or 1")
        rating = row.get("feedback")
        if rating is not None and (type(rating) is not int or rating not in (-1, 1)):
            raise ValueError("invalid feedback rating")
        if not (row["not_found"] or row["weak"] or rating == -1):
            continue
        at = row.get("at")
        if type(at) not in (float, int) or not math.isfinite(at) or at < 0 or at > 32503680000:
            raise ValueError("invalid event timestamp")
        key = query or ""
        item = grouped.setdefault(key, {"question": query or None, "occurrences": 0, "reasons": [],
                                       "suggestion": "Review or add an approved document answering this question."
                                       if query else "Collect a concrete example question before choosing a document."})
        item["occurrences"] += 1
        for reason, active in (("not found / not stated", row["not_found"]), ("low confidence", row["weak"]),
                               ("thumbs down", rating == -1)):
            if active and reason not in item["reasons"]:
                item["reasons"].append(reason)
    package = body.get("package", {})
    if not isinstance(package, dict):
        raise ValueError("package must be an object")  # noqa: TRY004 — malformed external JSON
    # Report identity is self-reported, not a signed package or certification.
    identity = {key: value[:200] for key in ("package_id", "name", "version")
                if isinstance(value := package.get(key), str)}
    return {"package": identity, "imported_at": time.time(), "suggestions": list(grouped.values()),
            "note": "Untrusted operator report. Suggestions only; no sources, scope or index were changed."}


def save_report(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as stream:
        temp = Path(stream.name)
        try:
            json.dump(report, stream, ensure_ascii=False)
            stream.flush()
            os.fsync(stream.fileno())
            os.replace(temp, path)
        finally:
            temp.unlink(missing_ok=True)
