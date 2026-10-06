"""Data Prep Chat — server-side tool-calling chat inside the Data Prep page.

Lets the user drive training-data preparation through natural language. The
chat can use tools to inspect parsed sources, read their content, list
existing Q&A pairs, and create new ones. Works with both local models
(via the ModelManager provider system) and external OpenAI-compatible
APIs (base_url + api_key passed per request).

Tool calling strategy:
- External API (OpenAI-compatible): native `tools` parameter, the API
  returns structured `tool_calls`. Standard, works everywhere.
- Local model (ModelManager): the system prompt instructs the model to
  emit `<tool_call>{"name":"...","arguments":{...}}</tool_call>` blocks.
  We parse that text out of the completion. Works with any chat model
  regardless of native tool-call support.

The endpoint runs a server-side tool-calling loop (max 6 rounds) so the
model can chain: list_sources -> read_source -> create_qa_pairs.

CORS: not needed, same-origin.
"""
from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Callable
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from finetune_studio.data.fs.paths import project_dir
from finetune_studio.guide import toolcalls
from finetune_studio.guide import tools as guide_tools
from finetune_studio.guide.loop import authoritative_readiness_reply, run_guide_loop
from finetune_studio.guide.prompt import build_system_prompt
from finetune_studio.guide.tools import ToolContext

log = logging.getLogger(__name__)
router = APIRouter()

# Tool catalog, system prompt, tool dispatch and the loop live in ``finetune_studio.guide``
# (one definition for this route, the SSE route and the Guide panel). The names below stay
# importable for existing callers and tests.
TOOLS_CATALOG = guide_tools.TOOLS_CATALOG
SYSTEM_PROMPT = build_system_prompt(ToolContext())
MAX_TOOL_ROUNDS = 8
_strip_thinking = toolcalls.strip_thinking
_extract_tool_calls = toolcalls.extract_tool_calls
_strip_thinking_reply = toolcalls.strip_thinking_reply
_looks_truncated = toolcalls.looks_truncated
_TRUNCATION_MSG = toolcalls.TRUNCATION_MSG
_authoritative_readiness_reply = authoritative_readiness_reply


def _run_tool(pid: str, name: str, args: dict) -> dict:
    """Run one guide tool for project ``pid`` (kept for existing callers; UI events are dropped)."""
    result = guide_tools.run_tool(ToolContext(pid=pid), name, args)
    result.pop("ui_event", None)
    return result


def _messages_to_prompt(messages: list[dict]) -> tuple[str, list[dict]]:
    """Flatten OpenAI-style messages into a single prompt string for local
    models that don't take a structured messages list. Returns
    (system_prefix, rendered_messages) — we keep the system separate so
    the chat endpoint can prepend it without re-rendering user msgs."""
    sys_prefix = ""
    rendered: list[dict] = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        if role == "system":
            sys_prefix += content + "\n\n"
        else:
            rendered.append({"role": role, "content": content})
    return sys_prefix.strip(), rendered


def parse_gen(body: dict) -> dict[str, Any]:
    """Generation kwargs from a request body (``gen{}`` or top-level keys), clamped to llama-safe ranges."""
    gen_raw: dict[str, Any] = dict(body.get("gen") or {})
    for k in ("temperature", "max_tokens", "top_p", "top_k", "repeat_penalty"):
        if k in body and k not in gen_raw:
            gen_raw[k] = body[k]
    gen: dict[str, Any] = {}
    if "temperature" in gen_raw:
        try:
            gen["temperature"] = max(0.0, min(2.0, float(gen_raw["temperature"])))
        except (TypeError, ValueError):
            pass
    if "max_tokens" in gen_raw:
        try:
            gen["max_tokens"] = max(32, min(16384, int(gen_raw["max_tokens"])))
        except (TypeError, ValueError):
            pass
    if "top_p" in gen_raw:
        try:
            gen["top_p"] = max(0.0, min(1.0, float(gen_raw["top_p"])))
        except (TypeError, ValueError):
            pass
    if "top_k" in gen_raw:
        try:
            gen["top_k"] = max(0, int(gen_raw["top_k"]))
        except (TypeError, ValueError):
            pass
    if "repeat_penalty" in gen_raw:
        try:
            gen["repeat_penalty"] = max(0.5, min(2.0, float(gen_raw["repeat_penalty"])))
        except (TypeError, ValueError):
            pass
    gen.setdefault("temperature", 0.2)
    gen.setdefault("max_tokens", 4096)
    return gen


async def resolve_chat_backend(
    provider_id: str | None, external: dict | None
) -> tuple[dict | None, JSONResponse | None]:
    """Pick the chat backend (explicit provider, external API, or the configured helper).

    Returns ``(backend, None)`` or ``(None, error_response)``. Never silently reuses a model that
    is not the configured helper.
    """
    backend: dict | None = None
    if external:
        try:
            import httpx  # noqa: F401
        except ImportError:
            return None, JSONResponse({"error": "external_api requires httpx (install httpx)"}, status_code=500)
        backend = {
            "kind": "external",
            "base_url": (external.get("base_url") or "").rstrip("/"),
            "api_key": external.get("api_key") or "",
            "model_id": external.get("model_id") or "",
        }
        if not backend["base_url"] or not backend["model_id"]:
            return None, JSONResponse({"error": "external_api.base_url and model_id required"}, status_code=400)
    elif provider_id:
        # Explicit provider_id — only path that may call manager.load().
        # Lazy imports: data_prep_chat.py is imported by app.py during
        # router registration, so importing `inference_engine` (which is
        # created at app.py module import time) at the top of this file
        # would create a partial-module cycle. Import inside the handler.
        from finetune_studio.models.manager import get_manager
        from finetune_studio.webui.app import inference_engine

        mgr = get_manager()
        cfg = mgr.get_provider(provider_id)
        if not cfg:
            return None, JSONResponse({"error": f"unknown provider {provider_id}"}, status_code=404)

        # Fast path: the global inference engine already holds THIS model.
        # Use it directly to avoid a duplicate Llama() instance racing with
        # mmap on the same GGUF file. The manager load() can wedge a busy
        # host in that scenario (CPU pinned, status API hangs).
        # NOTE: provider config uses `model_id` as the field name, not
        # `model_path`. The provider's value is usually absolute; the
        # engine's value is whatever was passed to chat-v2/load (often
        # relative). Normalise both to absolute before comparing.
        import os
        engine_path = getattr(inference_engine, "model_path", None) or ""
        provider_path = cfg.get("model_id") or cfg.get("model_path") or ""

        def _abs(p: str) -> str:
            if not p:
                return ""
            if os.path.isabs(p):
                return os.path.normpath(p)
            # Resolve relative to the project root (cwd of the uvicorn process).
            return os.path.normpath(os.path.join(os.getcwd(), p))

        engine_abs = _abs(engine_path)
        provider_abs = _abs(provider_path)
        if (
            getattr(inference_engine, "model", None) is not None
            and engine_abs
            and engine_abs == provider_abs
        ):
            log.info(
                "data-prep chat: using global engine for provider %s "
                "(path=%s); skipping manager load to avoid duplicate Llama",
                provider_id, engine_abs,
            )
            backend = {
                "kind": "global",
                "engine": inference_engine,
                "provider_id": provider_id,
            }
        else:
            log.info(
                "data-prep chat: engine_path=%s != provider_path=%s; "
                "falling through to manager.load",
                engine_abs, provider_abs,
            )
            # The engine holds a DIFFERENT model than the provider wants —
            # free it first. Otherwise it stays resident alongside whatever
            # manager.load() below loads, both competing for VRAM.
            if getattr(inference_engine, "model", None) is not None:
                await asyncio.to_thread(inference_engine.unload)
            try:
                await asyncio.to_thread(mgr.load, provider_id)
            except Exception as e:
                log.exception("failed to load provider %s", provider_id)
                return None, JSONResponse({"error": f"failed to load provider: {e}"}, status_code=500)
            backend = {"kind": "provider", "manager": mgr, "provider_id": provider_id}
    else:
        # No provider_id: require the configured GGUF helper — never
        # silently reuse a different Inference/manager model (e.g. a merged
        # project LoRA). Explicit provider_id / external_api remain allowed.
        from finetune_studio.data.prep.generator import (
            helper_resolution_error,
            resolve_helper_backend,
        )

        loaded = resolve_helper_backend()
        if loaded is None:
            return None, JSONResponse({"error": helper_resolution_error()}, status_code=409)
        backend = dict(loaded)
        log.info(
            "data-prep chat: provider_id omitted; using helper %s backend (%s)",
            backend.get("helper_label"),
            backend.get("kind"),
        )
    return backend, None


def make_chat_fn(backend: dict, gen: dict) -> Callable[[list[dict]], str]:
    """One blocking model call for the resolved backend."""
    if backend["kind"] == "external":
        return lambda msgs: _chat_external(backend, msgs, gen)
    if backend["kind"] == "global":
        return lambda msgs: _chat_global_engine(backend, msgs, gen)
    # raise_errors: a dead helper must surface as an error, not as an empty reply
    return lambda msgs: _chat_local(backend, msgs, gen, raise_errors=True)


@router.post("/projects/{pid}/data-prep/chat")
async def data_prep_chat(pid: str, request: Request):
    """Server-side chat with tool calling (JSON response; the Guide panel uses ``/api/guide/chat`` SSE).

    Body:
      messages: list[dict]  — OpenAI-style [{role, content}, ...]
      provider_id: str      — optional; when set, load/use that ModelManager provider
                              (mutually exclusive with external_api). When omitted the
                              configured helper must be loaded; never auto-switches models.
      external_api: dict    — {base_url, api_key, model_id, name?} for OpenAI-compatible endpoints
      max_rounds: int       — cap tool-call loop iterations (default 8, max 12)
      gen: dict / top-level — temperature, max_tokens, top_p, top_k, repeat_penalty

    Returns:
      {ok, reply, tool_calls, rounds, backend[, ui_events, forced_final]}
    """
    body = await request.json()
    if not isinstance(body, dict):
        return JSONResponse({"error": "JSON object body required"}, status_code=400)
    messages: list[dict] = body.get("messages") or []
    if not messages or not isinstance(messages, list):
        return JSONResponse({"error": "messages required"}, status_code=400)
    provider_id: str | None = body.get("provider_id")
    external: dict | None = body.get("external_api")
    if external is not None and not isinstance(external, dict):
        return JSONResponse({"error": "external_api must be an object"}, status_code=400)
    try:
        max_rounds = max(1, min(int(body.get("max_rounds") or MAX_TOOL_ROUNDS), 12))
    except (TypeError, ValueError):
        return JSONResponse({"error": "max_rounds must be an integer"}, status_code=400)
    gen = parse_gen(body)

    # Validate the backend before touching the filesystem so a missing project doesn't mask a
    # backend-config error (and vice versa).
    backend, error = await resolve_chat_backend(provider_id, external)
    if error is not None or backend is None:
        return error
    if not project_dir(pid).exists():
        return JSONResponse({"error": f"project {pid} not found"}, status_code=404)

    from finetune_studio import db
    project = db.get_project(pid) or {}
    ctx = ToolContext(pid=pid, page="chat")
    chat_fn = make_chat_fn(backend, gen)
    events = await asyncio.to_thread(lambda: list(run_guide_loop(
        ctx, messages, chat_fn, max_rounds=max_rounds, project_name=project.get("name"),
        max_tokens=gen.get("max_tokens", 4096),
    )))
    tool_calls = [{k: ev[k] for k in ("name", "arguments", "result")} for ev in events if ev["type"] == "tool_result"]
    backend_name = "external" if backend["kind"] == "external" else "provider"
    last = events[-1]
    if last["type"] == "error":
        if last.get("truncated"):
            return {"ok": False, "error": last["error"], "reply": last["error"], "tool_calls": tool_calls,
                    "rounds": last["rounds"], "backend": backend_name}
        return JSONResponse({"error": last["error"]}, status_code=last.get("status", 502))
    return {
        "ok": True,
        "reply": last["reply"],
        "tool_calls": tool_calls,
        "rounds": last["rounds"],
        "backend": backend_name,
        "forced_final": last["forced_final"],
        "ui_events": [ev["event"] for ev in events if ev["type"] == "ui"],
    }


def _chat_external(backend: dict, messages: list[dict], gen: dict | None = None) -> str:
    """One round of chat via an OpenAI-compatible HTTP endpoint.

    We use the native `tools` parameter so the model returns structured
    tool_calls; the message we build for the next round mirrors what the
    OpenAI SDK does.
    """
    import httpx
    base = backend["base_url"]
    url = base + "/chat/completions"
    g = gen or {}
    payload = {
        "model": backend["model_id"],
        "messages": messages,
        "tools": [{"type": "function", "function": t} for t in TOOLS_CATALOG],
        "tool_choice": "auto",
        "temperature": g.get("temperature", 0.2),
        "max_tokens": g.get("max_tokens", 4096),
        "top_p": g.get("top_p", 0.9),
    }
    headers = {"Content-Type": "application/json"}
    if backend["api_key"]:
        headers["Authorization"] = f"Bearer {backend['api_key']}"
    resp = httpx.post(url, json=payload, headers=headers, timeout=120.0)
    if resp.status_code >= 400:
        raise RuntimeError(f"external api {resp.status_code}: {resp.text[:300]}")
    data = resp.json()
    choice = (data.get("choices") or [{}])[0]
    msg = choice.get("message") or {}
    # If the model returned native tool_calls, synthesize a `<tool_call>` block
    # so the rest of the loop (parsing + execution) is backend-agnostic.
    if msg.get("tool_calls"):
        for tc in msg["tool_calls"]:
            fn = tc.get("function") or {}
            name = fn.get("name", "")
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {}
            block = json.dumps({"name": name, "arguments": args})
            return f"<tool_call>{block}</tool_call>"
    return msg.get("content") or ""


def _chat_global_engine(backend: dict, messages: list[dict], gen: dict | None = None) -> str:
    """One round of chat via the global inference engine (the single
    Llama() instance the app keeps loaded at any time).

    Used when the manager would otherwise create a duplicate Llama on the
    same GGUF path — that path wedges the service in mmap. The global
    engine already holds the model; we just call its create_chat_completion
    with the same gen params a manager provider would.

    The global engine uses llama-cpp-python's auto chat-template (chatml
    for Qwen3, etc.), so we hand it the messages as-is including the
    leading system prompt — no manual fold-in needed.
    """
    engine = backend["engine"]
    model = getattr(engine, "model", None)
    if model is None:
        raise RuntimeError("global engine has no model loaded")

    g = gen or {}
    kwargs: dict = {
        "messages": messages,
        "temperature": g.get("temperature", 0.2),
        "max_tokens": g.get("max_tokens", 4096),
        "top_p": g.get("top_p", 0.9),
    }
    # Optional gen params that llama-cpp-python accepts.
    for k in ("top_k", "repeat_penalty", "stop", "seed", "stream"):
        if k in g:
            kwargs[k] = g[k]

    resp = model.create_chat_completion(**kwargs)
    # llama-cpp-python returns a dict with 'choices': [{'message': {'content': ...}}]
    try:
        return resp["choices"][0]["message"]["content"] or ""
    except (KeyError, IndexError, TypeError):
        # Defensive: if the schema is unusual, return the raw dict string.
        return str(resp)


def _chat_local(backend: dict, messages: list[dict], gen: dict | None = None, *, raise_errors: bool = False) -> str:
    """One round of chat via the local ModelManager provider.

    Uses `mgr.chat()` (which calls llama_cpp.create_chat_completion) so the
    model's proper chat template is applied. This is critical for Qwen-style
    instruct models: a manually-rendered prompt (role.upper() + colons) often
    produces empty output because the model expects its ChatML tokens. Falling
    back to generate() for any provider that doesn't expose chat().
    """
    mgr = backend["manager"]
    g = gen or {}
    _temp = g.get("temperature", 0.2)
    _max  = g.get("max_tokens", 4096)
    _topp = g.get("top_p", 0.9)
    # Build the messages list we send to llama_cpp: drop the leading system
    # message because llama_cpp.create_chat_completion takes it via the
    # `messages` array (system role is supported there). For local chat models
    # we also rely on the system prompt's instruction to emit
    # <tool_call>{...}</tool_call> blocks when the model wants to act; the
    # native tool-call API is only used for external OpenAI-compat backends.
    chat_messages: list[dict] = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        if not content:
            continue
        # Local chat templates usually require assistant / user roles only —
        # system messages get folded into the first user turn's prefix so
        # models that don't grok the system role (older GGUF quants) still
        # see the rules.
        if role == "system" and not chat_messages:
            chat_messages.append({"role": "user", "content": content})
            continue
        if role == "system":
            # Append to last user message as a "system reminder".
            for j in range(len(chat_messages) - 1, -1, -1):
                if chat_messages[j].get("role") == "user":
                    chat_messages[j] = {
                        **chat_messages[j],
                        "content": chat_messages[j]["content"] + "\n\n" + content,
                    }
                    break
            else:
                chat_messages.append({"role": "user", "content": content})
            continue
        chat_messages.append({"role": role, "content": content})

    # Drop trailing duplicate user / empty assistant turns that llama_cpp
    # may reject.
    cleaned: list[dict] = []
    for m in chat_messages:
        if not m.get("content"):
            continue
        cleaned.append(m)
    if not cleaned:
        return ""

    # Ensure conversation starts with a user turn (required by most chat
    # templates).
    if cleaned[0]["role"] != "user":
        cleaned = [{"role": "user", "content": "(start)"}] + cleaned

    # Append a final nudge so instruction-tuned models actually produce a
    # reply (some chat templates leave the model hanging after tool
    # descriptions otherwise).
    if cleaned[-1]["role"] == "assistant":
        cleaned.append({"role": "user", "content": "(continue)"})
    try:
        text = mgr.chat(
            cleaned,
            max_tokens=_max,
            temperature=_temp,
            top_p=_topp,
        )
    except Exception as chat_error:
        # Fall back to generate() with a flattened prompt for providers that
        # only support raw text-completion (rare).
        log.exception("manager.chat failed; falling back to generate()")
        sys_prefix, rendered = _messages_to_prompt(messages)
        parts = []
        if sys_prefix:
            parts.append(sys_prefix)
        for m in rendered:
            parts.append(f"{m['role'].upper()}: {m['content']}")
        parts.append("ASSISTANT:")
        prompt = "\n\n".join(parts)
        try:
            text = mgr.generate(
                prompt,
                max_tokens=_max,
                temperature=_temp,
                top_p=_topp,
            )
        except Exception:
            log.exception("manager.generate fallback also failed")
            if raise_errors:
                raise chat_error from None   # the first failure is the real cause
            text = ""
    return text or ""


@router.get("/projects/{pid}/data-prep/chat/tools")
async def list_tools(pid: str):
    """Catalog of tools the chat exposes to the model (useful for debugging
    + for UI to show a help panel)."""
    from finetune_studio import db
    if not db.get_project(pid):
        return JSONResponse({"error": "project not found"}, status_code=404)
    return {"tools": TOOLS_CATALOG}
