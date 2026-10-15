"""``RemoteEngine``: the in-process face of the inference worker, same interface as ``InferenceEngine``.

Attributes callers read (``model``, ``model_path``, ``n_ctx``, ``offload``, ``_busy``, ``_last_used``, ...) come
from a status snapshot fetched over the socket and cached for a moment. Nothing here spawns a worker
except :meth:`load` (through ``launch.ensure_worker``); reads against a dead socket mean "no model".
"""
from __future__ import annotations

import http.client
import json
import time
from pathlib import Path
from typing import Any, ClassVar

from finetune_studio.supervisor.client import UnixHTTPConnection
from finetune_studio.supervisor.paths import inference_socket_path

STATUS_TTL_S = 0.5
STATUS_TIMEOUT_S = 5.0
LOAD_TIMEOUT_S = 3600.0
GENERATE_TIMEOUT_S = 3600.0
UNLOAD_TIMEOUT_S = 300.0


class WorkerUnavailable(RuntimeError):
    """No inference worker answers on the socket."""


class EngineCrashed(RuntimeError):
    """The worker died while serving this call. The model is gone; it is NOT reloaded automatically."""


_EMPTY: dict[str, Any] = {
    "loaded": False, "model_path": None, "is_gguf": False, "vision": False, "mmproj_path": None,
    "n_ctx": None, "n_gpu_layers": None, "offload": {}, "idle_seconds": 0, "busy": 0, "last_used": 0.0,
    "loading_path": None, "loading_started": 0.0, "pid": None,
}


class RemoteModel:
    """What ``engine.model`` returns while a model is loaded: truthy, and the one direct-model call we route."""

    def __init__(self, engine: RemoteEngine) -> None:
        self._engine = engine

    def create_chat_completion(self, **kwargs: Any) -> dict[str, Any]:
        return self._engine._call("POST", "/v1/chat_completion", kwargs, GENERATE_TIMEOUT_S, "chat completion")


class RemoteEngine:
    def __init__(self, socket_path: Path | None = None) -> None:
        self._socket = socket_path
        self._cache: dict[str, Any] | None = None
        self._cache_at = 0.0
        self.last_error: str = ""
        # ``inference_engine.tokenizer`` is only ever assigned by the in-process engine; keep the name for parity.
        self.tokenizer = None

    @property
    def socket_path(self) -> Path:
        return self._socket or inference_socket_path()

    # ── transport ───────────────────────────────────────────────────

    def _call(self, method: str, url: str, body: dict[str, Any] | None, timeout: float, what: str) -> Any:
        conn = UnixHTTPConnection(str(self.socket_path), timeout)
        try:
            payload = json.dumps(body).encode("utf-8") if body is not None else None
            headers = {"Content-Type": "application/json"} if payload is not None else {}
            try:
                conn.request(method, url, body=payload, headers=headers)
                resp = conn.getresponse()
                raw = resp.read()
            except (ConnectionRefusedError, FileNotFoundError) as exc:
                raise WorkerUnavailable(f"no inference worker on {self.socket_path}: {exc}") from exc
            except (http.client.RemoteDisconnected, ConnectionResetError, BrokenPipeError, http.client.BadStatusLine) as exc:
                self._invalidate()
                raise EngineCrashed(self._crash_message(what)) from exc
            except TimeoutError as exc:
                raise RuntimeError(f"inference worker did not answer the {what} within {timeout:.0f} s") from exc
        finally:
            conn.close()
        try:
            data = json.loads(raw or b"{}")
        except ValueError:
            data = {"error": raw.decode("utf-8", "replace")[:300]}
        if resp.status >= 400:
            raise RuntimeError(str(data.get("error") or f"HTTP {resp.status}"))
        return data

    def _crash_message(self, what: str) -> str:
        from finetune_studio.inference.launch import describe_worker_exit

        why = describe_worker_exit()
        return (f"the inference worker died during the {what}{why}. The model is unloaded; "
                "load it again to retry (a model that crashes the worker is not reloaded automatically).")

    def _invalidate(self) -> None:
        self._cache, self._cache_at = None, 0.0

    def _status(self, fresh: bool = False) -> dict[str, Any]:
        now = time.monotonic()
        if not fresh and self._cache is not None and now - self._cache_at < STATUS_TTL_S:
            return self._cache
        try:
            snap = self._call("GET", "/v1/status", None, STATUS_TIMEOUT_S, "status")
            self.last_error = ""
        except (WorkerUnavailable, EngineCrashed, RuntimeError) as exc:
            self.last_error = str(exc)
            snap = dict(_EMPTY)
        self._cache, self._cache_at = snap, time.monotonic()
        return snap

    # ── InferenceEngine interface ───────────────────────────────────
    # Status-backed attributes resolve through __getattr__ (only consulted when the instance has no such
    # attribute), so tests and callers can still do ``engine.model = fake`` / ``patch.object(engine, "n_ctx", 1)``
    # exactly as with the in-process engine: an instance attribute shadows the live status until deleted.
    _STATUS_ATTRS: ClassVar[dict[str, tuple[str, Any]]] = {
        "model_path": ("model_path", None), "is_gguf": ("is_gguf", False), "vision": ("vision", False),
        "mmproj_path": ("mmproj_path", None), "n_ctx": ("n_ctx", None), "n_gpu_layers": ("n_gpu_layers", None),
        "idle_seconds": ("idle_seconds", 0), "_busy": ("busy", 0), "_last_used": ("last_used", 0.0),
        "_loading_path": ("loading_path", None), "_loading_started": ("loading_started", 0.0),
    }

    def __getattr__(self, name: str) -> Any:
        if name == "model":
            return RemoteModel(self) if self._status().get("loaded") else None
        if name == "offload":
            return dict(self._status().get("offload") or {})
        if name in self._STATUS_ATTRS:
            key, default = self._STATUS_ATTRS[name]
            value = self._status().get(key)
            return default if value is None else value
        raise AttributeError(name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "model" and value is None:  # ``engine.model = None`` meant "unloaded" for the old engine
            self.__dict__.pop("model", None)
            self.unload()
            return
        object.__setattr__(self, name, value)

    def load(self, model_path: str, **kwargs: Any) -> None:
        from finetune_studio.inference.launch import ensure_worker
        from finetune_studio.smart_brain.model_cache import release_brain_models

        # The RAG embedder/reranker live in THIS process and share the GPU: free them before the worker loads.
        release_brain_models("model load")
        ensure_worker(self.socket_path)
        self._invalidate()
        body = {"model_path": str(model_path), **{k: v for k, v in kwargs.items() if v is not None}}
        try:
            snap = self._call("POST", "/v1/load", body, LOAD_TIMEOUT_S, f"load of {Path(str(model_path)).name}")
        finally:
            self._invalidate()
        self._cache, self._cache_at = snap, time.monotonic()

    def unload(self) -> None:
        try:
            self._call("POST", "/v1/unload", {}, UNLOAD_TIMEOUT_S, "unload")
        except WorkerUnavailable:
            pass  # nothing is running, so nothing is loaded
        finally:
            self._invalidate()

    def generate(self, messages: list[dict[str, Any]], max_tokens: int = 1024, temperature: float = 0.7, top_p: float = 0.9,
                 top_k: int = 40, repeat_penalty: float = 1.1, stop: list[str] | None = None, think: bool = False) -> str:
        body = {"messages": messages, "max_tokens": max_tokens, "temperature": temperature, "top_p": top_p, "top_k": top_k,
                "repeat_penalty": repeat_penalty, "stop": stop, "think": think}
        try:
            data = self._call("POST", "/v1/generate", body, GENERATE_TIMEOUT_S, "generation")
        except WorkerUnavailable as exc:
            raise RuntimeError("No model loaded") from exc
        finally:
            self._invalidate()
        return str(data.get("text", ""))

    # Model-file helpers read headers locally; no process boundary needed.
    @staticmethod
    def estimate_memory(model_path: str, **kw: Any) -> dict[str, Any]:
        from finetune_studio.testing.inference import InferenceEngine
        return InferenceEngine.estimate_memory(model_path, **kw)

    @staticmethod
    def read_model_metadata(model_path: str) -> dict[str, Any]:
        from finetune_studio.testing.inference import InferenceEngine
        return InferenceEngine.read_model_metadata(model_path)

    def worker_info(self) -> dict[str, Any]:
        """Where the model process is and how it is doing (for ``/api/inference/status``)."""
        from finetune_studio.inference.launch import worker_info
        return worker_info(self._status(fresh=True))
