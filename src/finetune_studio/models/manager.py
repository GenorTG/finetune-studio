"""ModelManager — one inference model loaded at a time.

Embeddings have their own resident model (sentence-transformers) that is
NEVER unloaded. The inference model is swapped atomically: a request to
load a new local model first unloads the current one.
"""

from __future__ import annotations

import logging
import os
import sqlite3
import threading
import time
from pathlib import Path

from finetune_studio.models.providers import (
    ModelProvider,
    ProviderConfig,
    build_provider,
)

log = logging.getLogger(__name__)

_DB_PATH = Path(os.environ.get("FTS_DB", str(Path.home() / ".finetune-studio" / "fts.db")))


def _ensure_db() -> None:
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(_DB_PATH) as c:
        c.executescript("""
        CREATE TABLE IF NOT EXISTS model_providers (
            id          TEXT PRIMARY KEY,
            name        TEXT NOT NULL,
            kind        TEXT NOT NULL,
            model_id    TEXT NOT NULL DEFAULT '',
            base_url    TEXT NOT NULL DEFAULT '',
            api_key     TEXT NOT NULL DEFAULT '',
            extra_json  TEXT NOT NULL DEFAULT '{}',
            created_at  REAL NOT NULL
        );
        CREATE TABLE IF NOT EXISTS app_state (
            key   TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        """)
        # Default local helper (27B GGUF) if no providers configured yet.
        from finetune_studio.models.helper import (
            DEFAULT_HELPER_EXTRA,
            DEFAULT_HELPER_LABEL,
            DEFAULT_HELPER_PROVIDER_ID,
            default_helper_gguf_path,
        )

        cur = c.execute("SELECT COUNT(*) FROM model_providers").fetchone()
        if cur[0] == 0:
            now = time.time()
            c.execute(
                "INSERT INTO model_providers (id, name, kind, model_id, base_url, api_key, extra_json, created_at) "
                "VALUES (?,?,?,?,?,?,?,?)",
                (
                    DEFAULT_HELPER_PROVIDER_ID,
                    DEFAULT_HELPER_LABEL,
                    "local_gguf",
                    default_helper_gguf_path(),
                    "",
                    "",
                    json_dumps(dict(DEFAULT_HELPER_EXTRA)),
                    now,
                ),
            )
        else:
            # Rename legacy "Local GGUF" label to the explicit helper name.
            c.execute(
                "UPDATE model_providers SET name = ? "
                "WHERE id = ? AND (name = '' OR name = 'Local GGUF' OR name = 'local')",
                (DEFAULT_HELPER_LABEL, DEFAULT_HELPER_PROVIDER_ID),
            )


_LOG = logging.getLogger(__name__)


# Loader parameters exposed through ModelManager.load(pid, extra=...).
# Kept module-level so callers (load() merge loop, tests) can iterate.
_LOADER_KEYS = (
    "n_ctx", "n_gpu_layers", "n_batch", "n_threads",
    "seed", "rope_freq_base", "rope_freq_scale",
    "flash_attn", "mmap", "mlock", "keep_in_memory",
)


def json_dumps(d: dict) -> str:
    import json
    return json.dumps(d, ensure_ascii=False)


def json_loads(s: str) -> dict:
    import json
    try:
        return json.loads(s) if s else {}
    except (TypeError, json.JSONDecodeError):
        return {}


class ModelManager:
    """Holds the single active inference provider.

    State transitions:
      idle --load(cfg)--> loading (unloads prior local) --> loaded
      loaded --unload()--> idle
    """

    def __init__(self):
        self._lock = threading.RLock()
        # Dedicated lock for *model invocation* — llama_cpp.Llama is not thread-safe,
        # so concurrent chat/generate calls must be serialized. Remote providers are
        # safe either way, so we always go through this lock; cost is negligible.
        self._invoke_lock = threading.Lock()
        self._provider: ModelProvider | None = None
        self._active_id: str = ""
        _ensure_db()

    # ── provider CRUD (DB) ─────────────────────────────────────

    def list_providers(self) -> list[dict]:
        from finetune_studio.models.helper import annotate_provider

        with sqlite3.connect(_DB_PATH) as c:
            rows = c.execute("SELECT id, name, kind, model_id, base_url, api_key, extra_json, created_at FROM model_providers ORDER BY created_at").fetchall()
        out = []
        for r in rows:
            d = {
                "id": r[0], "name": r[1], "kind": r[2],
                "model_id": r[3], "base_url": r[4],
                "api_key_set": bool(r[5]),
                "api_key": r[5] if r[5] else "",  # returned for editing; never log it
                "extra": json_loads(r[6]),
                "created_at": r[7],
            }
            out.append(annotate_provider(d))
        return out

    def get_provider(self, pid: str) -> dict | None:
        for p in self.list_providers():
            if p["id"] == pid:
                return p
        return None

    def upsert_provider(self, **kw) -> dict:
        allowed = {"id", "name", "kind", "model_id", "base_url", "api_key", "extra"}
        fields = {k: v for k, v in kw.items() if k in allowed}
        if "id" not in fields or not fields["id"]:
            raise ValueError("Provider id required")
        with sqlite3.connect(_DB_PATH) as c:
            existing = c.execute("SELECT id FROM model_providers WHERE id = ?", (fields["id"],)).fetchone()
            if existing:
                sets = ", ".join(f"{k} = ?" for k in fields if k != "id")
                vals = [fields[k] for k in fields if k != "id"] + [fields["id"]]
                c.execute(f"UPDATE model_providers SET {sets} WHERE id = ?", vals)
            else:
                now = time.time()
                cols = ["id", "name", "kind", "model_id", "base_url", "api_key", "extra_json", "created_at"]
                vals = [fields.get("id"), fields.get("name", ""), fields.get("kind", "openai_compat"),
                        fields.get("model_id", ""), fields.get("base_url", ""), fields.get("api_key", ""),
                        json_dumps(fields.get("extra", {})), now]
                c.execute(f"INSERT INTO model_providers ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})", vals)
        return self.get_provider(fields["id"])

    def delete_provider(self, pid: str) -> bool:
        with sqlite3.connect(_DB_PATH) as c:
            cur = c.execute("DELETE FROM model_providers WHERE id = ?", (pid,))
            return cur.rowcount > 0

    # ── active model management ──────────────────────────────────

    def active(self) -> dict | None:
        with self._lock:
            if self._provider is None:
                return None
            d = self._provider.describe()
            d["idle_seconds"] = int(time.time() - self._provider._loaded_at) if self._provider._loaded_at else 0
            return d

    def load(self, pid: str, extra: dict | None = None) -> dict:
        cfg_row = self.get_provider(pid)
        if not cfg_row:
            raise ValueError(f"Unknown provider: {pid}")
        # Merge any runtime-provided loader params on top of the persisted
        # extra JSON. None / unknown keys are dropped so the provider's own
        # defaults kick in. This lets every page (data-prep, inference,
        # ...) load the same provider with different n_ctx / n_gpu_layers /
        # n_batch / n_threads / seed / rope / flash_attn / mmap / mlock /
        # keep_in_memory without a DB write.
        merged_extra = dict(cfg_row.get("extra") or {})
        if extra:
            for k in _LOADER_KEYS:
                if k in extra and extra[k] is not None:
                    merged_extra[k] = extra[k]
        with self._lock:
            if (self._provider is not None
                    and self._active_id == pid
                    and self._provider.is_loaded()):
                # Fast path: same provider, already loaded with the same
                # merged_extra — no-op. Without this guard, calling load()
                # a second time was building a fresh LocalGGUFProvider,
                # calling .load() on it, and triggering 'Failed to load
                # model from file' because a second Llama() can't be
                # instantiated while the first still holds the mmap/VRAM.
                if getattr(self._provider, "_used_extra", None) == merged_extra:
                    return self.active()
                # Loader params actually changed — unload + reload so the
                # new n_ctx / n_gpu_layers / etc. take effect.
                self._safe_unload()
        cfg = ProviderConfig(
            id=cfg_row["id"], name=cfg_row["name"], kind=cfg_row["kind"],
            model_id=cfg_row["model_id"], base_url=cfg_row["base_url"],
            api_key=cfg_row.get("api_key", ""),
            extra=merged_extra,
        )
        new_provider = build_provider(cfg)
        new_provider._used_extra = dict(merged_extra)  # snapshot for fast-path
        with self._lock:
            # Unload previous local model (mutual exclusion)
            if self._provider is not None and isinstance(self._provider, type(new_provider)) is False:
                # Different kind — unload unconditionally
                self._safe_unload()
            elif self._provider is not None and self._active_id != pid:
                self._safe_unload()
            new_provider.load()
            self._provider = new_provider
            self._active_id = pid
        return self.active()

    def unload(self) -> None:
        with self._lock:
            self._safe_unload()
            self._active_id = ""

    def _safe_unload(self) -> None:
        if self._provider is None:
            return
        try:
            self._provider.unload()
        except Exception as e:  # noqa: BLE001
            log.warning("unload failed: %s", e)
        self._provider = None

    def chat(self, messages: list[dict], **gen) -> str:
        with self._lock:
            p = self._provider
        if p is None:
            raise RuntimeError("No model loaded")
        with self._invoke_lock:
            return p.chat(messages, **gen)

    def generate(self, prompt: str, **gen) -> str:
        with self._lock:
            p = self._provider
        if p is None:
            raise RuntimeError("No model loaded")
        with self._invoke_lock:
            return p.generate(prompt, **gen)


# Singleton (process-wide)
_manager: ModelManager | None = None
_manager_lock = threading.Lock()


def get_manager() -> ModelManager:
    global _manager
    with _manager_lock:
        if _manager is None:
            _manager = ModelManager()
        return _manager
