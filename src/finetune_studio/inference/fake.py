"""A stand-in for ``InferenceEngine`` with the same attributes: used by the worker's ``--engine fake``
test seam so the process/protocol/crash paths are tested without a GPU or a model file.

Triggers: loading a path ending in ``.crash`` aborts the process (native-abort stand-in); a user
message ``__crash__`` aborts mid-generate; ``__sleep__:N`` generates for N seconds.
"""
from __future__ import annotations

import os
import threading
import time
from typing import Any


class FakeModel:
    def __init__(self, engine: FakeEngine) -> None:
        self._engine = engine

    def create_chat_completion(self, **kwargs: Any) -> dict[str, Any]:
        text = self._engine.generate(kwargs.get("messages") or [], max_tokens=int(kwargs.get("max_tokens", 16)))
        return {"id": "fake", "object": "chat.completion", "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}]}


class FakeEngine:
    def __init__(self) -> None:
        self.model: FakeModel | None = None
        self.tokenizer = None
        self.model_path: str | None = None
        self.is_gguf = False
        self.vision = False
        self.mmproj_path: str | None = None
        self.n_ctx: int | None = None
        self.n_gpu_layers: int | None = None
        self.offload: dict[str, Any] = {}
        self._last_used = 0.0
        self._busy = 0
        self._busy_lock = threading.Lock()
        self._loading_path: str | None = None
        self._loading_started = 0.0

    def load(self, model_path: str, **kw: Any) -> None:
        self._loading_path, self._loading_started = model_path, time.time()
        try:
            if model_path.endswith(".crash"):
                os.abort()
            if model_path.endswith(".missing"):
                raise FileNotFoundError(f"model not found: {model_path}")
            time.sleep(0.05)
            self.model = FakeModel(self)
            self.model_path = model_path
            self.is_gguf = model_path.endswith(".gguf")
            self.n_ctx = int(kw.get("n_ctx") or 0) or 4096
            self.n_gpu_layers = int(kw.get("n_gpu_layers", -1))
            self.offload = {"mode": "gpu", "gpu_layers": 32, "total_layers": 32, "warnings": []}
            self._last_used = time.time()
        finally:
            self._loading_path = None

    def unload(self) -> None:
        self.model = None
        self.model_path = None
        self.is_gguf = False
        self.n_ctx = self.n_gpu_layers = None
        self.offload = {}
        self._last_used = 0.0

    @property
    def idle_seconds(self) -> int:
        return 0 if self.model is None or not self._last_used else int(time.time() - self._last_used)

    def generate(self, messages: list[dict[str, Any]], max_tokens: int = 1024, **kw: Any) -> str:
        if self.model is None:
            raise RuntimeError("No model loaded")
        with self._busy_lock:
            self._busy += 1
        try:
            content = str((messages[-1] if messages else {}).get("content", ""))
            if content == "__crash__":
                os.abort()
            if content.startswith("__sleep__:"):
                time.sleep(float(content.split(":", 1)[1]))
            self._last_used = time.time()
            return f"echo: {content}"[: max(1, max_tokens * 8)]
        finally:
            with self._busy_lock:
                self._busy -= 1
