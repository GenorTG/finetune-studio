"""Canonical artifact display names (Genor's rule 2026-09-20).

Bare run-dir names like ``6f64c46a/merged`` or a running model labelled
``_merged`` are unreadable. Every artifact gets a FULL name:

    <Project> · <Base model>[ · v<N> <label>][ · <Kind>][ · <QUANT>][ · abliterated]

e.g. ``E2E Smoke 20260920 · Qwen3-4B · v1 e2e-smoke-v1 · merged``
     ``Vaelindrath · Qwen3-4B · GGUF Q5_K_M``
     ``Qwen3.8-27B · Q4_K_M · abliterated``

Pure string helpers + one DB-aware resolver (``display_for_path``). The
resolver reads the app DB through ``finetune_studio.db`` — never a hardcoded
sqlite path (the old raw-path lookup silently failed on the deployed box
because the DB lives at FTS_DB_PATH, which is why names fell back to hashes).
"""
from __future__ import annotations

import json
import logging
import re

log = logging.getLogger(__name__)

# q4_k_m, Q5_K_M, q8_0, q4.0, f16/bf16/fp16/fp8
_QUANT_RE = re.compile(
    r"(?i)(?<![a-z0-9])(q\d+(?:[._][a-z0-9]+){1,2}|bf16|fp16|fp8|f16)(?![a-z0-9])"
)
_ABLIT_RE = re.compile(r"(?i)abliterat")
# projects/<pid>/runs/<rid>/<kind> — pid/rid are hex-ish ids (8-16 chars)
_RUN_PATH_RE = re.compile(
    r"projects[/\\]([0-9a-f]{6,})[/\\]runs[/\\]([0-9a-f]{6,})[/\\]([A-Za-z0-9_.\-]+)"
)
_HF_CACHE_RE = re.compile(r"models--([^/\\]+)--([^/\\]+)")

_GENERIC_KINDS = {
    "merged": "merged",
    "abliterated": "merged",
    "adapter": "LoRA adapter",
    "gguf": "GGUF",
    "gptq": "GPTQ",
    "export": "export",
    "model": "",
}


_EXT_RE = re.compile(r"(?i)\.(gguf|safetensors|bin|pt|onnx|ckpt)$")


def detect_quant(name_or_path: str) -> str | None:
    # strip the trailing extension first: "model-q8_0.gguf" must not let the
    # regex swallow ".gguf" as a second quant token (→ Q8_0.GGUF)
    s = _EXT_RE.sub("", (name_or_path or "").rstrip("/\\"))
    m = _QUANT_RE.search(s)
    if not m:
        return None
    q = m.group(1).upper()
    return {"FP16": "F16"}.get(q, q)


def detect_abliterated(name_or_path: str) -> bool:
    return bool(_ABLIT_RE.search(name_or_path or ""))


def short_base(name_or_path: str) -> str:
    """``Qwen/Qwen3-4B`` / ``models--Qwen--Qwen3-4B/snapshots/<hash>`` /
    ``…/hf_models/Qwen__Qwen3-4B`` → ``Qwen3-4B`` (quant/abliteration tokens
    stripped — they belong in the modifier tail, not the base name)."""
    s = (name_or_path or "").strip().rstrip("/\\")
    if not s:
        return ""
    m = _HF_CACHE_RE.search(s)
    if m:
        s = m.group(2)
    else:
        tail = re.split(r"[/\\]", s)[-1]
        if "__" in tail and not s.lower().endswith(".gguf"):
            tail = tail.split("__", 1)[1]
        if tail.lower() in ("snapshots", "hub", "models") or _looks_hash(tail):
            parts = re.split(r"[/\\]", s)
            tail = next((p for p in reversed(parts) if p and not _looks_hash(p)), tail)
        s = tail
    s = re.sub(r"\.(gguf|safetensors|bin|pt)$", "", s, flags=re.IGNORECASE)
    s = _QUANT_RE.sub("", s)
    s = _ABLIT_RE.sub("", s)
    s = re.sub(r"[-_]{1,2}$", "", s)
    s = re.sub(r"^(unsloth|TheBloke|bartowski|mraderman)[-_]", "", s, flags=re.IGNORECASE)
    return s.strip() or (name_or_path or "").strip()


def _looks_hash(seg: str) -> bool:
    return len(seg) >= 16 and all(c in "0123456789abcdef" for c in seg.lower())


def kind_label(dirname: str) -> str:
    return _GENERIC_KINDS.get((dirname or "").lower(), (dirname or "").strip())


def model_full_name(
    project: str = "",
    base: str = "",
    version: str = "",
    kind: str = "",
    quant: str | None = None,
    abliterated: bool = False,
) -> str:
    parts = [p for p in (project.strip(), base.strip(), version.strip(), kind.strip()) if p]
    if quant:
        parts.append(quant)
    if abliterated:
        parts.append("abliterated")
    return " · ".join(parts)


def _version_label_for_run(project_id: str, run_id: str) -> str:
    """``v3 my-label`` if a pinned project version references this run id."""
    try:
        from finetune_studio import db

        for v in db.list_versions(project_id):
            manifest = v.get("manifest") or v.get("manifest_json") or "{}"
            if isinstance(manifest, str):
                try:
                    manifest = json.loads(manifest)
                except ValueError:
                    continue
            for entry in manifest.get("training_runs") or []:
                rid = str(entry.get("run_id") or "")
                if rid and (rid.startswith(run_id) or run_id.startswith(rid)):
                    label = str(v.get("label") or "").strip()
                    num = v.get("version_number") or v.get("number") or ""
                    return f"v{num} {label}".strip()
    except Exception:  # noqa: BLE001  # display helper must never raise
        return ""
    return ""


def resolve_run_path(path: str) -> dict | None:
    """Parse ``projects/<pid>/runs/<rid>/<kind>`` and enrich from the DB."""
    m = _RUN_PATH_RE.search(path or "")
    if not m:
        return None
    pid, rid, kind_dir = m.group(1), m.group(2), m.group(3)
    out: dict = {"project_id": pid, "run_id": rid, "kind_dir": kind_dir}
    project_name = ""
    base = ""
    try:
        from finetune_studio import db

        proj = db.get_project(pid)
        if proj:
            project_name = str(proj.get("name") or "")
        for r in db.list_runs(project_id=pid):
            full = str(r.get("id") or "")
            if full.startswith(rid) or rid.startswith(full[:8]):
                base = short_base(str(r.get("base_model") or ""))
                out["run_id_full"] = full
                break
    except Exception:  # display resolver must never raise
        log.debug("naming: run lookup failed for %s", path, exc_info=True)
    out["project_name"] = project_name
    out["base"] = base
    out["kind"] = kind_label(kind_dir)
    out["quant"] = detect_quant(path)
    out["abliterated"] = detect_abliterated(path)
    out["version"] = _version_label_for_run(pid, rid)
    return out


def display_for_path(path: str, size_hint: str = "") -> str:
    """Best human-readable name for a model path (run export or plain dir)."""
    info = resolve_run_path(path)
    if info:
        name = model_full_name(
            project=info.get("project_name", ""),
            base=info.get("base", ""),
            version=info.get("version", ""),
            kind=info.get("kind", ""),
            quant=info.get("quant"),
            abliterated=info.get("abliterated", False),
        )
        if name:
            return name
    tail = re.split(r"[/\\]", (path or "").rstrip("/\\"))[-1] or path or ""
    base = short_base(tail)
    quant = detect_quant(tail)
    abl = detect_abliterated(tail)
    parts = [p for p in (base or tail, quant, "abliterated" if abl else "") if p]
    if size_hint and parts:
        parts.append(size_hint)
    return " · ".join(parts) if parts else (path or "")
