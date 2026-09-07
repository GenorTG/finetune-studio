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

import json
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Optional

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
        self._n_ctx: int = int(config.extra.get("n_ctx", 4096))
        self._n_gpu_layers: int = int(config.extra.get("n_gpu_layers", 99))

    def load(self) -> None:
        from llama_cpp import Llama
        log.info("LocalGGUFProvider loading %s (n_ctx=%d, n_gpu=%d)", self.config.model_id, self._n_ctx, self._n_gpu_layers)
        with self._lock:
            self._llama = Llama(
                model_path=self.config.model_id,
                n_ctx=self._n_ctx,
                n_gpu_layers=self._n_gpu_layers,
                n_batch=512,
                mmap=True,
                flash_attn=True,
                verbose=False,
            )
        self._loaded_at = time.time()

    def unload(self) -> None:
        with self._lock:
            if self._llama is not None:
                try:
                    del self._llama
                except Exception:
                    pass
                self._llama = None
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        self._loaded_at = 0.0
        log.info("LocalGGUFProvider unloaded")

    def is_loaded(self) -> bool:
        return self._llama is not None

    def _gen_kwargs(self, gen: dict) -> dict:
        return dict(
            max_tokens=int(gen.get("max_tokens", 1024)),
            temperature=float(gen.get("temperature", 0.7)),
            top_p=float(gen.get("top_p", 0.9)),
            top_k=int(gen.get("top_k", 40)),
            repeat_penalty=float(gen.get("repeat_penalty", 1.1)),
        )

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
            # Fallback to chat mode
            return self.chat([{"role": "user", "content": prompt}], **gen)


def build_provider(config: ProviderConfig) -> ModelProvider:
    if config.kind == "local_gguf":
        return LocalGGUFProvider(config)
    if config.kind == "openai_compat":
        return OpenAICompatProvider(config)
    raise ValueError(f"Unknown provider kind: {config.kind}")


# ── Catalog of well-known providers (used to populate the picker UI) ────

PROVIDER_PRESETS: list[dict] = [
    {"id": "local", "name": "Local GGUF", "kind": "local_gguf", "model_id": "", "base_url": "", "api_key": ""},
    {"id": "openai", "name": "OpenAI", "kind": "openai_compat", "base_url": "https://api.openai.com/v1", "model_id": "gpt-4o-mini", "api_key": ""},
    {"id": "openrouter", "name": "OpenRouter", "kind": "openai_compat", "base_url": "https://openrouter.ai/api/v1", "model_id": "anthropic/claude-3.5-sonnet", "api_key": ""},
    {"id": "opencode-go", "name": "opencode-go", "kind": "openai_compat", "base_url": "https://api.opencode.ai/v1", "model_id": "default", "api_key": ""},
    {"id": "anthropic", "name": "Anthropic (via openai-compat proxy)", "kind": "openai_compat", "base_url": "https://api.anthropic.com/v1", "model_id": "claude-3-5-sonnet-latest", "api_key": ""},
    {"id": "MiniMax", "name": "MiniMax", "kind": "openai_compat", "base_url": "https://api.MiniMax.ai/v1", "model_id": "MiniMax/MiniMax-M3", "api_key": ""},
    {"id": "custom", "name": "Custom OpenAI-compat URL", "kind": "openai_compat", "base_url": "", "model_id": "", "api_key": ""},
]
