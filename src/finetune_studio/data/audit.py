"""Evidence-grade audits for uploaded files, curated data, and evaluations.

The audit is intentionally deterministic.  It never asks a model whether a
file was parsed correctly; it hashes the stored bytes, reparses those bytes,
and checks the persisted artefacts and provenance links.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from finetune_studio.data import project_filesystem as pfs
from finetune_studio.data.fs.paths import file_dir
from finetune_studio.data.parsers import parse_bytes
from finetune_studio.data.prep.chunker import chunk_text
from finetune_studio.data.prep.qa_validate import content_tokens, token_overlap_ratio


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _source_raw_path(pid: str, source: dict[str, Any]) -> Path:
    sha = str(source.get("sha256") or "")
    filename = str(source.get("filename") or source.get("name") or "upload")
    return file_dir(pid, sha) / filename


def audit_source(pid: str, source: dict[str, Any]) -> dict[str, Any]:
    """Audit one persisted source from raw bytes through chunk artefacts."""
    raw_path = _source_raw_path(pid, source)
    expected_sha = str(source.get("sha256") or "")
    result: dict[str, Any] = {
        "source_id": source.get("id", ""),
        "filename": source.get("filename", ""),
        "expected_sha256": expected_sha,
        "raw_path": str(raw_path),
        "ok": False,
        "errors": [],
        "warnings": [],
    }
    if not raw_path.is_file():
        result["errors"].append("raw_file_missing")
        return result
    raw = raw_path.read_bytes()
    actual_sha = _sha256(raw)
    result.update({"byte_count": len(raw), "actual_sha256": actual_sha})
    if expected_sha and actual_sha != expected_sha:
        result["errors"].append("raw_hash_mismatch")

    parsed_path = file_dir(pid, actual_sha) / "parsed.txt"
    if not parsed_path.is_file():
        result["errors"].append("parsed_text_missing")
        return result
    parsed = parsed_path.read_text(encoding="utf-8", errors="replace")
    reparsed = parse_bytes(str(source.get("filename") or raw_path.name), raw)
    expected_text = str(reparsed.get("text") or "")
    result.update({
        "parsed_char_count": len(parsed),
        "parsed_sha256": _sha256(parsed.encode("utf-8")),
        "reparsed_char_count": len(expected_text),
        "reparsed_sha256": _sha256(expected_text.encode("utf-8")),
        "parser": reparsed.get("metadata", {}).get("parser", ""),
        "parser_warnings": reparsed.get("metadata", {}).get("warnings", []),
    })
    if parsed != expected_text:
        result["errors"].append("persisted_parse_differs_from_deterministic_reparse")
    if not parsed.strip():
        result["errors"].append("empty_parsed_text")

    chunks_dir = file_dir(pid, actual_sha) / "chunks"
    chunks = [p.read_text(encoding="utf-8", errors="replace")
              for p in sorted(chunks_dir.glob("*.txt"))] if chunks_dir.is_dir() else []
    expected_chunks = chunk_text(parsed)
    result.update({
        "chunk_count": len(chunks),
        "expected_chunk_count": len(expected_chunks),
        "chunk_char_count": sum(len(c) for c in chunks),
    })
    if chunks != expected_chunks:
        result["errors"].append("persisted_chunks_differ_from_deterministic_chunking")
    if not chunks:
        result["errors"].append("no_chunks")
    source_tokens = content_tokens(parsed)
    chunk_tokens: set[str] = set()
    for chunk in chunks:
        chunk_tokens.update(content_tokens(chunk))
    result["token_coverage_ratio"] = round(
        token_overlap_ratio(source_tokens, chunk_tokens), 4
    ) if source_tokens else 0.0
    if source_tokens and not chunk_tokens:
        result["errors"].append("chunks_have_no_content_tokens")
    result["ok"] = not result["errors"]
    return result


def audit_project_sources(pid: str) -> dict[str, Any]:
    """Audit every QA source and its raw/file manifest."""
    sources = pfs.list_qa_sources(pid)
    audits = [audit_source(pid, source) for source in sources]
    return {
        "project_id": pid,
        "source_count": len(sources),
        "passed": sum(1 for item in audits if item["ok"]),
        "failed": sum(1 for item in audits if not item["ok"]),
        "status": "not_applicable" if not sources else ("pass" if all(item["ok"] for item in audits) else "fail"),
        "sources": audits,
    }


def audit_qa_pairs(pid: str, *, exported_path: str | None = None) -> dict[str, Any]:
    """Check pair provenance, chunk coverage, grounding, and export loss."""
    sources = {s.get("id", ""): s for s in pfs.list_qa_sources(pid)}
    pairs = pfs.list_qa_pairs(pid, status="approved")
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    errors: list[dict[str, Any]] = []
    for pair in pairs:
        sid = str(pair.get("source_id") or "")
        by_source[sid].append(pair)
        source = sources.get(sid)
        if source is None:
            errors.append({"id": pair.get("id"), "error": "unknown_source_id"})
            continue
        chunk_idx = int(pair.get("chunk_idx") or 0)
        if chunk_idx < 1 or chunk_idx > int(source.get("chunk_count") or 0):
            errors.append({"id": pair.get("id"), "error": "chunk_index_out_of_range"})
        chunk_text_value = str(pair.get("chunk_text") or "")
        answer = str(pair.get("answer") or "")
        if chunk_text_value and answer and not content_tokens(answer) & content_tokens(chunk_text_value):
            errors.append({"id": pair.get("id"), "error": "answer_has_no_chunk_token_overlap"})

    source_coverage = {}
    for sid, source in sources.items():
        rows = by_source.get(sid, [])
        covered = sorted({int(row.get("chunk_idx") or 0) for row in rows})
        total = int(source.get("chunk_count") or 0)
        source_coverage[sid] = {
            "filename": source.get("filename", ""),
            "pairs": len(rows),
            "chunks_total": total,
            "chunks_covered": len([i for i in covered if i > 0]),
            "uncovered_chunks": [i for i in range(1, total + 1) if i not in covered],
        }

    export_count = None
    export_errors: list[str] = []
    if exported_path:
        path = Path(exported_path)
        if not path.is_file():
            export_errors.append("export_missing")
        else:
            export_count = 0
            for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                if not line.strip():
                    continue
                try:
                    json.loads(line)
                    export_count += 1
                except json.JSONDecodeError:
                    export_errors.append(f"invalid_jsonl_line:{line_no}")
            if export_count != len(pairs):
                export_errors.append("approved_pair_count_differs_from_export_count")

    return {
        "project_id": pid,
        "approved_pair_count": len(pairs),
        "source_count": len(sources),
        "source_coverage": source_coverage,
        "errors": errors + [{"error": e} for e in export_errors],
        "export_count": export_count,
        "passed": bool(pairs) and not errors and not export_errors,
        "status": "not_applicable" if not pairs else ("pass" if not errors and not export_errors else "fail"),
        "rejected_or_pending_count": len(pfs.list_qa_pairs(pid)) - len(pairs),
    }


def audit_suite_cases(suite_path: str, dataset_path: str | None = None) -> dict[str, Any]:
    """Audit suite completeness and metadata against its source dataset."""
    path = Path(suite_path)
    result: dict[str, Any] = {"suite_path": str(path), "errors": [], "ok": False}
    if not path.is_file():
        result["errors"].append("suite_missing")
        return result
    data = json.loads(path.read_text(encoding="utf-8"))
    meta: dict = {}
    if isinstance(data, dict):
        meta = data.get("meta") or {}
    cases = data.get("cases", []) if isinstance(data, dict) else data
    if not isinstance(cases, list):
        result["errors"].append("suite_cases_missing")
        return result
    names = [str(c.get("name", "")) for c in cases]
    result["case_count"] = len(cases)
    result["coverage"] = meta.get("coverage", "full")
    result["duplicate_names"] = sorted(k for k, v in Counter(names).items() if v > 1)
    if result["duplicate_names"]:
        result["errors"].append("duplicate_case_names")
    if dataset_path:
        rows = [json.loads(line) for line in Path(dataset_path).read_text(encoding="utf-8").splitlines() if line.strip()]
        result["dataset_count"] = len(rows)
        if result["coverage"] == "sampled":
            # A sampled suite is explicit opt-in; the audit checks its honesty
            # instead: the file must carry matching sample metadata.
            if int(meta.get("case_count", -1)) != len(cases) or int(meta.get("dataset_count", -1)) != len(rows):
                result["errors"].append("sampled_suite_meta_mismatch")
        elif len(rows) != len(cases):
            result["errors"].append("suite_dataset_count_mismatch")
    result["source_ids"] = sorted({str(c.get("source_id")) for c in cases if c.get("source_id")})
    result["ok"] = not result["errors"]
    return result
