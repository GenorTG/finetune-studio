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

from pydantic import BaseModel, Field, field_validator

from finetune_studio.data.rag_portable.constants import DEFAULT_TOP_K

log = logging.getLogger(__name__)

SETTINGS_KEY = "rag_export"
PROJECTS_KEY = "rag_export_projects"
RUNTIME_CONFIG_NAME = "rag.config.json"


class RagExportConfig(BaseModel):
    """Defaults baked into an exported package (and into its rag.config.json)."""

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
        if k not in RagExportConfig.model_fields:
            continue
        try:
            RagExportConfig(**{**out, k: v})
        except ValueError:
            continue
        out[k] = v
    return out


def load_defaults() -> RagExportConfig:
    return RagExportConfig(**_clean(_read().get(SETTINGS_KEY)))


def save_defaults(values: dict[str, Any]) -> RagExportConfig:
    cfg = RagExportConfig(**{**load_defaults().model_dump(), **values})
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
        RagExportConfig(**{**load_defaults().model_dump(), **values})  # validate
        cleaned = {k: v for k, v in values.items() if k in RagExportConfig.model_fields}
    if cleaned:
        projects[pid] = cleaned
    else:
        projects.pop(pid, None)
    data[PROJECTS_KEY] = projects
    _write(data)
    return cleaned


def effective_config(pid: str | None = None,
                     overrides: dict[str, Any] | None = None) -> RagExportConfig:
    merged = load_defaults().model_dump()
    if pid:
        merged.update(load_project_override(pid))
    merged.update({k: v for k, v in (overrides or {}).items() if v is not None})
    return RagExportConfig(**merged)
