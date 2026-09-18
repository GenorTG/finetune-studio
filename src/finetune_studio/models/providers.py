"""Model provider abstraction + ModelManager.

The WebUI doesn't talk to a single hardcoded model. Instead, it talks to
a `ModelProvider` (Local GGUF or any OpenAI-compatible remote endpoint).
The `ModelManager` enforces one local model loaded at a time — when a new
local model is requested, the previous one is unloaded first. Embeddings
have their own resident model that is never unloaded.

Providers are configured in `model_providers` table and stored as plain rows;
API keys live in that table (or env vars for server-side use).
"""

from __future__ import annotations

import contextlib
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger(__name__)


# ── Provider descriptor ─────────────────────────────────────────────────

@dataclass
class ProviderConfig:
    id: str
    name: str
    kind: str  # "local_gguf" | "openai_compat"
    model_id: str = ""  # local: filesystem path; remote: model name
    base_url: str = ""  # remote only
    api_key: str = ""  # remote only
    extra: dict[str, Any] = field(default_factory=dict)


# ── Abstract provider ──────────────────────────────────────────────────

class ModelProvider:
    """Base class for inference backends.

    A provider owns ONE model and exposes:
      - load(): bring model into memory / session
      - unload(): release
      - chat(messages, **gen): OpenAI-style chat completion, returns str
      - generate(prompt, **gen): raw text completion (for non-chat Q&A)
    """

    def __init__(self, config: ProviderConfig):
        self.config = config
        self._loaded_at: float = 0.0
        self._lock = threading.Lock()

    def load(self) -> None:
        raise NotImplementedError

    def unload(self) -> None:
        raise NotImplementedError

    def is_loaded(self) -> bool:
        return self._loaded_at > 0.0

    def chat(self, messages: list[dict], **gen) -> str:
        raise NotImplementedError

    def generate(self, prompt: str, **gen) -> str:
        raise NotImplementedError

    def describe(self) -> dict:
        return {
            "kind": self.config.kind,
            "id": self.config.id,
            "name": self.config.name,
            "model_id": self.config.model_id,
            "loaded": self.is_loaded(),
        }


# ── Local GGUF provider ────────────────────────────────────────────────

class LocalGGUFProvider(ModelProvider):
    """Backed by llama-cpp-python. Mutually exclusive — only one instance at a time."""

    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        self._llama = None  # the llama_cpp.Llama instance
        self._n_ctx: int = int(config.extra.get("n_ctx", 16384))
        # n_gpu_layers: -1 = all layers (llama.cpp native idiom). Legacy 99
        # is translated by ModelManager.load() before reaching here.
        self._n_gpu_layers: int = int(config.extra.get("n_gpu_layers", -1))
        self._n_batch: int = int(config.extra.get("n_batch", 512))
        self._n_threads: int = int(config.extra.get("n_threads", 0))
        self._seed: int = int(config.extra.get("seed", -1))
        self._rope_freq_base: float = float(config.extra.get("rope_freq_base", 0.0))
        self._rope_freq_scale: float = float(config.extra.get("rope_freq_scale", 0.0))
        self._flash_attn: bool = bool(config.extra.get("flash_attn", True))
        self._mmap: bool = bool(config.extra.get("mmap", True))
        self._mlock: bool = bool(config.extra.get("mlock", False))
        self._kv_type_k: int = int(config.extra.get("type_k", 0) or 0)
        self._kv_type_v: int = int(config.extra.get("type_v", 0) or 0)
        # Real layer count from the GGUF header (None for non-GGUF paths).
        from finetune_studio.models.gguf_layers import resolve_block_count
        self._topology = resolve_block_count(self.config.model_id)

    def load(self) -> None:
        from llama_cpp import Llama
        log.info(
            "LocalGGUFProvider loading %s (n_ctx=%d, n_gpu=%d, n_batch=%d, "
            "n_threads=%d, seed=%d, rope_base=%s, rope_scale=%s, "
            "flash=%s, mmap=%s, mlock=%s)",
            self.config.model_id, self._n_ctx, self._n_gpu_layers,
            self._n_batch, self._n_threads, self._seed,
            self._rope_freq_base, self._rope_freq_scale,
            self._flash_attn, self._mmap, self._mlock,
        )
        kwargs = {
            "model_path": self.config.model_id,
            "n_ctx": self._n_ctx,
            "n_gpu_layers": self._n_gpu_layers,
            "n_batch": self._n_batch,
            "mmap": self._mmap,
            "flash_attn": self._flash_attn,
            "verbose": False,
        }
        if self._n_threads > 0:
            kwargs["n_threads"] = self._n_threads
        if self._seed >= 0:
            kwargs["seed"] = self._seed
        if self._mlock:
            kwargs["use_mlock"] = True
        if self._rope_freq_base > 0:
            kwargs["rope_freq_base"] = self._rope_freq_base
        if self._rope_freq_scale > 0:
            kwargs["rope_freq_scale"] = self._rope_freq_scale
        if self._kv_type_k > 0:
            kwargs["type_k"] = self._kv_type_k
        if self._kv_type_v > 0:
            kwargs["type_v"] = self._kv_type_v
        if self._topology.get("block_count") is not None:
            log.info(
                "LocalGGUFProvider topology: %d transformer blocks, "
                "native ctx %s — n_gpu_layers=%d",
                self._topology["block_count"],
                self._topology.get("context_length"), self._n_gpu_layers,
            )
        with self._lock:
            self._llama = Llama(**kwargs)
        self._loaded_at = time.time()

    def describe(self) -> dict:
        d = {
            "kind": "local_gguf",
            "id": self.config.id,
            "name": self.config.name,
            "model_id": self.config.model_id,
            "loaded": self._llama is not None,
            "n_ctx": self._n_ctx,
            "n_gpu_layers": self._n_gpu_layers,
            "n_batch": self._n_batch,
        }
        # Real topology when the model is a GGUF (UI shows "36/36").
        block_count = self._topology.get("block_count")
        if block_count is not None:
            d["block_count"] = block_count
            d["context_length_native"] = self._topology.get("context_length")
            if self._n_gpu_layers == -1:
                d["gpu_layers_on"] = block_count
                d["gpu_layers_total"] = block_count
            elif self._n_gpu_layers > 0:
                d["gpu_layers_on"] = min(self._n_gpu_layers, block_count)
                d["gpu_layers_total"] = block_count
        return d

    def unload(self) -> None:
        with self._lock:
            if self._llama is not None:
                # llama-cpp free can raise on a partially-init handle; drop ref anyway.
                with contextlib.suppress(Exception):
                    del self._llama
                self._llama = None
        with contextlib.suppress(ImportError, AttributeError, RuntimeError):
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        self._loaded_at = 0.0
        log.info("LocalGGUFProvider unloaded")

    def is_loaded(self) -> bool:
        return self._llama is not None

    def _gen_kwargs(self, gen: dict) -> dict:
        return {
            "max_tokens": int(gen.get("max_tokens", 1024)),
            "temperature": float(gen.get("temperature", 0.7)),
            "top_p": float(gen.get("top_p", 0.9)),
            "top_k": int(gen.get("top_k", 40)),
            "repeat_penalty": float(gen.get("repeat_penalty", 1.1)),
        }

    def chat(self, messages: list[dict], **gen) -> str:
        if self._llama is None:
            raise RuntimeError("Local model not loaded")
        kwargs = self._gen_kwargs(gen)
        result = self._llama.create_chat_completion(messages=messages, **kwargs)
        content = result["choices"][0]["message"]["content"]
        return content.strip() if isinstance(content, str) else str(content).strip()

    def generate(self, prompt: str, **gen) -> str:
        if self._llama is None:
            raise RuntimeError("Local model not loaded")
        kwargs = self._gen_kwargs(gen)
        out = self._llama(prompt, **kwargs)
        return out["choices"][0]["text"].strip()


# ── OpenAI-compat provider ─────────────────────────────────────────────

class OpenAICompatProvider(ModelProvider):
    """Any service speaking the OpenAI chat-completions API."""

    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        self._session = None  # requests.Session

    def _client(self):
        if self._session is None:
            import requests
            self._session = requests.Session()
            if self.config.api_key:
                self._session.headers["Authorization"] = f"Bearer {self.config.api_key}"
        return self._session

    def load(self) -> None:
        # No-op for remote — just mark loaded; tokens counted by the upstream
        self._loaded_at = time.time()
        log.info("OpenAICompatProvider %s ready (model=%s, url=%s)", self.config.id, self.config.model_id, self.config.base_url)

    def unload(self) -> None:
        self._session = None
        self._loaded_at = 0.0
        log.info("OpenAICompatProvider %s closed", self.config.id)

    def chat(self, messages: list[dict], **gen) -> str:
        url = (self.config.base_url or "https://api.openai.com/v1").rstrip("/") + "/chat/completions"
        body = {
            "model": self.config.model_id,
            "messages": messages,
            "max_tokens": int(gen.get("max_tokens", 1024)),
            "temperature": float(gen.get("temperature", 0.7)),
            "top_p": float(gen.get("top_p", 0.9)),
        }
        if "stop" in gen:
            body["stop"] = gen["stop"]
        r = self._client().post(url, json=body, timeout=300)
        r.raise_for_status()
        data = r.json()
        return data["choices"][0]["message"]["content"].strip()

    def generate(self, prompt: str, **gen) -> str:
        # Most providers support /completions as well; if not, use chat with single user msg
        url = (self.config.base_url or "https://api.openai.com/v1").rstrip("/") + "/completions"
        body = {
            "model": self.config.model_id,
            "prompt": prompt,
            "max_tokens": int(gen.get("max_tokens", 1024)),
            "temperature": float(gen.get("temperature", 0.7)),
            "top_p": float(gen.get("top_p", 0.9)),
        }
        try:
            r = self._client().post(url, json=body, timeout=300)
            r.raise_for_status()
            return r.json()["choices"][0]["text"].strip()
        except Exception:
            # Many OpenAI-compat hosts lack /completions; fall back to chat.
            log.exception("completions failed; falling back to chat")
            return self.chat([{"role": "user", "content": prompt}], **gen)


def build_provider(config: ProviderConfig) -> ModelProvider:
    if config.kind == "local_gguf":
        return LocalGGUFProvider(config)
    if config.kind == "openai_compat":
        return OpenAICompatProvider(config)
    raise ValueError(f"Unknown provider kind: {config.kind}")


# ── Catalog of well-known providers (used to populate the picker UI) ────

def _local_helper_preset() -> dict:
    from finetune_studio.models.helper import (
        DEFAULT_HELPER_LABEL,
        DEFAULT_HELPER_PROVIDER_ID,
        default_helper_gguf_path,
    )

    return {
        "id": DEFAULT_HELPER_PROVIDER_ID,
        "name": DEFAULT_HELPER_LABEL,
        "kind": "local_gguf",
        "model_id": default_helper_gguf_path(),
        "base_url": "",
        "api_key": "",
    }


PROVIDER_PRESETS: list[dict] = [
    _local_helper_preset(),
    {"id": "openai", "name": "OpenAI", "kind": "openai_compat", "base_url": "https://api.openai.com/v1", "model_id": "gpt-4o-mini", "api_key": ""},
    {"id": "openrouter", "name": "OpenRouter", "kind": "openai_compat", "base_url": "https://openrouter.ai/api/v1", "model_id": "anthropic/claude-3.5-sonnet", "api_key": ""},
    {"id": "opencode-go", "name": "opencode-go", "kind": "openai_compat", "base_url": "https://api.opencode.ai/v1", "model_id": "default", "api_key": ""},
    {"id": "anthropic", "name": "Anthropic (via openai-compat proxy)", "kind": "openai_compat", "base_url": "https://api.anthropic.com/v1", "model_id": "claude-3-5-sonnet-latest", "api_key": ""},
    {"id": "MiniMax", "name": "MiniMax", "kind": "openai_compat", "base_url": "https://api.MiniMax.ai/v1", "model_id": "MiniMax/MiniMax-M3", "api_key": ""},
    {"id": "custom", "name": "Custom OpenAI-compat URL", "kind": "openai_compat", "base_url": "", "model_id": "", "api_key": ""},
]
