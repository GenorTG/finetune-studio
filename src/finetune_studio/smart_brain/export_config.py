"""Export-time defaults for the Portable RAG package.

Precedence when an export is built (low -> high):
built-in defaults < studio defaults (``settings.json`` key ``rag_export``)
< per-project override (``settings.json`` key ``rag_export_projects[<pid>]``)
< explicit request values.

Secrets are NEVER part of this model: the encryption passphrase is accepted
per export request and returned once; it is not stored here, in the DB, or in
logs.  The runtime counterpart is the shipped ``rag.config.json``.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from finetune_studio.smart_brain.constants import DEFAULT_TOP_K

log = logging.getLogger(__name__)

SETTINGS_KEY = "brain_export"
PROJECTS_KEY = "brain_export_projects"
LEGACY_KEYS = {"rag_export": SETTINGS_KEY, "rag_export_projects": PROJECTS_KEY}   # renamed 2026-10-14
RUNTIME_CONFIG_NAME = "rag.config.json"


class BrainExportConfig(BaseModel):
    """Defaults baked into an exported package (and into its rag.config.json).

    Search layers, each on its own: ``semantic_search`` (meaning search over the vectors; ships vectors.npy and,
    with ``include_models``, the embedder), ``keyword_search`` (BM25 word search; ships bm25.json); both on =
    hybrid, merged by Reciprocal Rank Fusion. ``reranker_enabled`` (a cross-encoder rescoring pass; needs the
    reranker bundled with ``include_models``). ``answer_enabled`` is the generation half of RAG: the server asks
    an OpenAI-compatible chat model (``llm_base_url`` / ``llm_model``; the key only at run time) to answer from
    the passages it found. Off = the server returns passages and the caller's own model writes the answer.
    """

    archive_format: Literal["tar", "tar.gz", "zip"] = "tar.gz"
    include_models: bool = False
    include_reranker: bool = True      # only meaningful with include_models
    reranker_enabled: bool = True      # runtime default; --no-reranker overrides
    device: str = "auto"               # auto (GPU first) | cpu | cuda | cuda:N | xpu | mps
    host: str = "127.0.0.1"            # non-loopback requires an auth token
    port: int = Field(8899, ge=1, le=65535)
    top_k: int = Field(DEFAULT_TOP_K, ge=1, le=100)
    encrypt: bool = True               # --no-encrypt is the explicit opt-out
    kdf_log_n: int = Field(17, ge=10, le=20)   # scrypt cost (N = 2**kdf_log_n)
    semantic_search: bool = True       # meaning search (vectors)
    keyword_search: bool = True        # word search (BM25)
    answer_enabled: bool = False       # RAG answers via an LLM endpoint
    llm_base_url: str = ""             # OpenAI-compatible base URL, e.g. http://127.0.0.1:11434/v1
    llm_model: str = ""
    llm_headers: dict[str, str] = Field(default_factory=dict)   # e.g. {"x-opencode-session": "{session}"}
    llm_body: dict[str, Any] = Field(default_factory=dict)      # merged into each request, e.g. reasoning_effort

    @field_validator("llm_headers")
    @classmethod
    def _llm_headers(cls, v: dict[str, str]) -> dict[str, str]:
        # A package is handed to other people: credentials never go in it (the key is RAG_LLM_API_KEY at run time).
        banned = [k for k in v if k.strip().lower() in ("authorization", "x-api-key", "api-key", "cookie")]
        if banned:
            raise ValueError(f"header {banned[0]!r} carries a credential; set RAG_LLM_API_KEY where the server runs")
        return {str(k).strip(): str(val) for k, val in v.items() if str(k).strip()}

    @model_validator(mode="after")
    def _layers(self) -> BrainExportConfig:
        if not (self.semantic_search or self.keyword_search):
            raise ValueError("turn on meaning search, word search or both: a package needs one way to find passages")
        return self

    @field_validator("llm_base_url")
    @classmethod
    def _llm_url(cls, v: str) -> str:
        v = v.strip().rstrip("/")
        if v and not v.startswith(("http://", "https://")):
            raise ValueError("llm_base_url must start with http:// or https://")
        return v

    @field_validator("device")
    @classmethod
    def _device(cls, v: str) -> str:
        import re
        if not re.fullmatch(r"(cpu|cuda(:\d+)?|xpu(:\d+)?|mps|auto)", v.strip()):
            raise ValueError("device must be auto, cpu, cuda, cuda:N, xpu, mps")
        return v.strip()

    @field_validator("host")
    @classmethod
    def _host(cls, v: str) -> str:
        v = v.strip()
        if not v or any(c.isspace() for c in v):
            raise ValueError("host must be a hostname or IP")
        return v

    def runtime_config(self) -> dict[str, Any]:
        """The subset shipped as ``rag.config.json`` (no secrets, no content)."""
        return {
            "device": self.device, "host": self.host, "port": self.port,
            "top_k": self.top_k, "reranker": self.reranker_enabled,
            "semantic": self.semantic_search, "keyword": self.keyword_search,
            "answer": self.answer_enabled, "llm_base_url": self.llm_base_url, "llm_model": self.llm_model.strip(),
            "llm_headers": dict(self.llm_headers), "llm_body": dict(self.llm_body),
        }


def _settings_path() -> Path:
    from finetune_studio.webui.routes import settings as s
    return s.SETTINGS_PATH


def _read() -> dict[str, Any]:
    p = _settings_path()
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text())
    except (OSError, ValueError) as e:
        log.warning("settings read failed: %s", e)
        return {}
    return data if isinstance(data, dict) else {}


def _write(data: dict[str, Any]) -> None:
    p = _settings_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(p)


def _clean(raw: Any) -> dict[str, Any]:
    """Keep only known, valid fields (a stale/invalid key never breaks export)."""
    if not isinstance(raw, dict):
        return {}
    out: dict[str, Any] = {}
    for k, v in raw.items():
        if k not in BrainExportConfig.model_fields:
            continue
        try:
            BrainExportConfig(**{**out, k: v})
        except ValueError:
            continue
        out[k] = v
    return out


def migrate_settings_keys() -> bool:
    """One-time rename of the export settings keys (``knowledge.migrate``). Returns whether anything moved."""
    data = _read()
    moved = False
    for old, new in LEGACY_KEYS.items():
        if old in data:
            data.setdefault(new, data[old])
            del data[old]
            moved = True
    if moved:
        _write(data)
    return moved


def load_defaults() -> BrainExportConfig:
    return BrainExportConfig(**_clean(_read().get(SETTINGS_KEY)))


def save_defaults(values: dict[str, Any]) -> BrainExportConfig:
    cfg = BrainExportConfig(**{**load_defaults().model_dump(), **values})
    data = _read()
    data[SETTINGS_KEY] = cfg.model_dump()
    _write(data)
    return cfg


def load_project_override(pid: str) -> dict[str, Any]:
    return _clean((_read().get(PROJECTS_KEY) or {}).get(pid))


def save_project_override(pid: str, values: dict[str, Any] | None) -> dict[str, Any]:
    """Persist a partial override for one project (``None``/{} clears it)."""
    data = _read()
    projects = data.get(PROJECTS_KEY)
    if not isinstance(projects, dict):
        projects = {}
    cleaned: dict[str, Any] = {}
    if values:
        BrainExportConfig(**{**load_defaults().model_dump(), **values})  # validate
        cleaned = {k: v for k, v in values.items() if k in BrainExportConfig.model_fields}
    if cleaned:
        projects[pid] = cleaned
    else:
        projects.pop(pid, None)
    data[PROJECTS_KEY] = projects
    _write(data)
    return cleaned


def effective_config(pid: str | None = None,
                     overrides: dict[str, Any] | None = None) -> BrainExportConfig:
    merged = load_defaults().model_dump()
    if pid:
        merged.update(load_project_override(pid))
    merged.update({k: v for k, v in (overrides or {}).items() if v is not None})
    return BrainExportConfig(**merged)
