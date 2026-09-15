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

import json
import logging
import re
import time
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from finetune_studio.data.fs import qa as qa_fs
from finetune_studio.data.fs.paths import project_dir

log = logging.getLogger(__name__)
router = APIRouter()

# Server-side tool catalog. Kept short and stable — the model prompt
# describes them in plain English so even a 7B model can pick the right one.
TOOLS_CATALOG = [
    {
        "name": "list_sources",
        "description": "List all parsed source files in this project. Returns each source's id, filename, status, and chunk count.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "read_source",
        "description": "Read the parsed text content of a source file by id. Use this AFTER list_sources to see what the file contains before generating Q&A pairs.",
        "parameters": {
            "type": "object",
            "properties": {"source_id": {"type": "string"}},
            "required": ["source_id"],
        },
    },
    {
        "name": "list_qa_pairs",
        "description": "List existing Q&A pairs in this project, optionally filtered by source_id or status (pending/approved/rejected).",
        "parameters": {
            "type": "object",
            "properties": {
                "source_id": {"type": "string"},
                "status": {"type": "string", "enum": ["pending", "approved", "rejected"]},
            },
        },
    },
    {
        "name": "create_qa_pairs",
        "description": "Create new Q&A pairs for a source. Each pair needs a source_id (from list_sources), a question, and an answer. Pairs land in 'pending' status so the user can review before approving.",
        "parameters": {
            "type": "object",
            "properties": {
                "source_id": {"type": "string"},
                "pairs": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "question": {"type": "string"},
                            "answer": {"type": "string"},
                        },
                        "required": ["question", "answer"],
                    },
                },
            },
            "required": ["source_id", "pairs"],
        },
    },
]

SYSTEM_PROMPT = """You are an expert training-data organizer for fine-tuning a language model. You help the user extract high-quality Q&A pairs from their parsed source files (markdown, text, PDF, code, OCR'd images, etc.) into a clean dataset.

# Rules

1. When the user mentions "the files" or "the data", ALWAYS call `list_sources` first to see what's available before generating anything.
2. Call `read_source` on each file you intend to mine, so the content is fresh in your context.
3. Generate Q&A pairs that test ACTUAL knowledge from the text — not generic questions. The answers should quote or closely paraphrase the source.
4. Aim for 3-8 pairs per source by default. Cover key facts, definitions, cause/effect, comparison, and applied reasoning.
5. Call `create_qa_pairs` with the full batch in ONE call, not one pair per call.
6. Be terse in prose — the data does the talking.

# Tool-call format (CRITICAL — follow exactly)

When you need to call a tool, output EXACTLY one tool_call block. Always close the tag:

<tool_call>{"name":"<tool_name>","arguments":{<json_args>}}</tool_call>

Closed examples (note the closing `</tool_call>` on its own line):

<tool_call>{"name":"list_sources","arguments":{}}</tool_call>

<tool_call>{"name":"read_source","arguments":{"source_id":"abc123"}}</tool_call>

<tool_call>{"name":"create_qa_pairs","arguments":{"source_id":"abc123","pairs":[{"question":"What is X?","answer":"X is ..."}]}}</tool_call>

When you have no more tool calls to make, respond with ONE short sentence summarising what you created. Do NOT wrap it in a tool_call tag.

# Hard rules

- ALWAYS close `<tool_call>` with `</tool_call>`. Never leave the tag open.
- Do NOT include chain-of-thought or reasoning in your reply. No "I need to..." or "Let me think about..." preambles. No `<think>` blocks.
- Emit EITHER one tool call OR a short summary. Never both at once.
- Do NOT echo the rules back to the user.
"""

TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
MAX_TOOL_ROUNDS = 6


def _run_tool(pid: str, name: str, args: dict) -> dict:
    """Execute a single tool against the project's filesystem. Returns a
    JSON-serializable dict."""
    try:
        if name == "list_sources":
            sources_dir = project_dir(pid) / "qa" / "sources"
            if not sources_dir.exists():
                return {"sources": []}
            out = []
            for p in sorted(sources_dir.glob("*.json")):
                try:
                    src = json.loads(p.read_text(encoding="utf-8"))
                except Exception:  # noqa: BLE001, S112
                    continue
                out.append({
                    "id": src.get("id", p.stem),
                    "filename": src.get("filename", "?"),
                    "status": src.get("status", "ready"),
                    "doc_count": src.get("doc_count", 1),
                    "chunk_count": src.get("chunk_count", 0),
                })
            return {"sources": out}
        if name == "read_source":
            sid = args.get("source_id", "")
            if not sid:
                return {"error": "source_id required"}
            src = qa_fs.read_qa_source(pid, sid)
            if not src:
                return {"error": f"source {sid} not found"}
            # Prefer the in-manifest text if present (legacy / future path).
            text = src.get("text") or src.get("parsed_text") or ""
            # The data-prep runner stores parsed text at
            # files/<sha256[:12]>/parsed.txt, NOT in the manifest. Fall
            # back to disk if the manifest has no text.
            if not text:
                sha = src.get("sha256") or sid
                from finetune_studio.data.fs.paths import file_dir
                parsed_path = file_dir(pid, sha) / "parsed.txt"
                if parsed_path.exists():
                    try:
                        text = parsed_path.read_text(encoding="utf-8", errors="replace")
                    except OSError:
                        text = ""
            return {
                "id": src.get("id", sid),
                "filename": src.get("filename", "?"),
                "text": text[:8000],
                "truncated": len(text) > 8000,
            }
        if name == "list_qa_pairs":
            pairs = qa_fs.list_qa_pairs(
                pid,
                source_id=args.get("source_id"),
                status=args.get("status"),
            )
            # Slim each pair for the model context.
            return {
                "pairs": [
                    {
                        "id": p.get("id"),
                        "source_id": p.get("source_id"),
                        "question": p.get("question", "")[:200],
                        "answer": p.get("answer", "")[:300],
                        "status": p.get("status", "pending"),
                    }
                    for p in pairs[:200]
                ],
                "count": len(pairs),
            }
        if name == "create_qa_pairs":
            sid = args.get("source_id", "")
            pairs = args.get("pairs") or []
            if not sid or not pairs:
                return {"error": "source_id and pairs required"}
            written = 0
            for pair in pairs:
                q = (pair.get("question") or "").strip()
                a = (pair.get("answer") or "").strip()
                if not q or not a:
                    continue
                qa_id = f"qa_{int(time.time()*1000)}_{written}"
                qa_fs.write_qa_pair(pid, {
                    "id": qa_id,
                    "source_id": sid,
                    "question": q,
                    "answer": a,
                    "status": "pending",
                    "created_at": time.time(),
                    "created_via": "data-prep-chat",
                })
                written += 1
            return {"written": written, "source_id": sid}
        return {"error": f"unknown tool: {name}"}
    except Exception as e:
        log.exception("tool %s failed", name)
        return {"error": f"tool {name} failed: {e}"}


def _strip_thinking(text: str) -> str:
    """Remove Qwen3 thinking blocks from model output.

    Handles both paired ``<think>…</think>`` and the common chat-template
    leak where only a bare closing ``</think>`` appears (opening tag was
    injected into the prompt, so the model never emits it).
    """
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    if "</think>" in cleaned:
        cleaned = cleaned.split("</think>", 1)[1]
    return cleaned


def _extract_tool_calls(text: str) -> list[dict]:
    """Pull `<tool_call>{...}</tool_call>` blocks out of a model reply.

    Robust against two recurring issues with local Qwen3 GGUF + llama-cpp:
      1. The model leaks chain-of-thought (Qwen3's native thinking-mode
         output). Strip `<think>...</think>` blocks BEFORE regex matching so
         they don't contaminate the tool-call JSON or appear in the visible
         reply. Also drop everything up to a bare ``</think>``.
      2. The model frequently emits `<tool_call>{...}` WITHOUT a closing
         `</tool_call>` tag. Try the strict closed form first; on miss,
         fall back to a brace-balanced extractor that walks the unmatched
         opening tag and grabs everything up to the first balanced `}`.
    """
    # 1. Strip Qwen3 thinking-mode blocks (paired + bare closing tag).
    cleaned = _strip_thinking(text)
    # 2. Strip any leading/trailing prose so the closing tag (or unclosed
    # block) is clearly delimited. We do NOT mutate the text that flows
    # into the visible reply (that's `_strip_thinking_reply`'s job).
    calls: list[dict] = []
    seen: set[tuple[str, str]] = set()  # dedupe by (name, json_args)

    def _try_parse(raw: str) -> None:
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            return
        name = obj.get("name")
        args = obj.get("arguments") or {}
        if not isinstance(name, str) or not isinstance(args, dict):
            return
        key = (name, json.dumps(args, sort_keys=True))
        if key in seen:
            return
        seen.add(key)
        calls.append({"name": name, "arguments": args})

    # 2a. Strict pass: properly-closed <tool_call>{...}</tool_call>.
    for m in TOOL_CALL_RE.finditer(cleaned):
        _try_parse(m.group(1))

    # 2b. Fallback pass: unclosed <tool_call>{...}  (Qwen3 occasionally
    # forgets the closing tag). Find every opening tag, then brace-balance
    # forward to find the end of the JSON object.
    if not calls:
        for m in re.finditer(r"<tool_call>\s*", cleaned):
            start = m.end()
            depth = 0
            in_string = False
            escape = False
            end = -1
            for i in range(start, len(cleaned)):
                ch = cleaned[i]
                if escape:
                    escape = False
                    continue
                if ch == "\\":
                    escape = True
                    continue
                if ch == '"':
                    in_string = not in_string
                    continue
                if in_string:
                    continue
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        end = i + 1
                        break
            if end > start:
                _try_parse(cleaned[start:end])
            # Stop after the first successful extraction — the parser
            # driver already enforces one-tool-call-per-round upstream.
            if calls:
                break

    return calls


def _strip_thinking_reply(text: str) -> str:
    """Strip thinking blocks from a model reply before it becomes the
    user-visible assistant message."""
    return _strip_thinking(text).strip()


_TRUNCATION_MSG = (
    "Response was cut off at max_tokens={n} — raise Max tokens and retry"
)


def _looks_truncated(text: str) -> bool:
    """True when generation likely hit max_tokens mid-tool-call or think."""
    if not text:
        return False
    # Unclosed think block (opening present, no closer).
    if "<think>" in text and "</think>" not in text:
        return True
    # Truncated tool call: opening tag present but we couldn't parse a call.
    return "<tool_call>" in text and not _extract_tool_calls(text)


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


@router.post("/projects/{pid}/data-prep/chat")
async def data_prep_chat(pid: str, request: Request):
    """Server-side chat with tool calling for organizing training data.

    Body:
      messages: list[dict]  — OpenAI-style [{role, content}, ...]
      provider_id: str      — optional; when set, load/use that ModelManager
                              provider (mutually exclusive with external_api).
                              When omitted, reuse an already-loaded model
                              (Inference engine preferred, else manager active).
                              Never auto-loads a second model.
      external_api: dict    — {base_url, api_key, model_id, name?} for
                              OpenAI-compatible endpoints
      max_rounds: int       — cap tool-call loop iterations (default 6)
      gen: dict             — generation kwargs merged with sensible defaults
                              (temperature, max_tokens, top_p, top_k,
                              repeat_penalty). Used for both backend paths.
      Top-level temperature/max_tokens/top_p/top_k also accepted (chat UI).

    Returns:
      {ok, reply, tool_calls, rounds}
    """
    body = await request.json()
    messages: list[dict] = body.get("messages") or []
    if not messages:
        return {"error": "messages required"}
    provider_id: str | None = body.get("provider_id")
    external: dict | None = body.get("external_api")
    max_rounds = max(1, min(int(body.get("max_rounds") or MAX_TOOL_ROUNDS), 12))

    # Generation kwargs — allow per-request override. Anything not supplied
    # falls back to sane defaults for tool-calling (low temperature, roomy
    # max_tokens for thinking models). Clamp so junk from the frontend
    # doesn't crash llama_cpp. Chat UI sends top-level keys; also accept gen{}.
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
            gen["max_tokens"] = max(32, min(8192, int(gen_raw["max_tokens"])))
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

    # Resolve the chat backend (provider OR external API OR already-loaded
    # model). Validate this before touching the filesystem so a missing
    # project doesn't mask a backend-config error (and vice versa).
    backend: dict | None = None
    if external:
        try:
            import httpx  # noqa: F401
        except ImportError:
            return {"error": "external_api requires httpx (install httpx)"}
        backend = {
            "kind": "external",
            "base_url": (external.get("base_url") or "").rstrip("/"),
            "api_key": external.get("api_key") or "",
            "model_id": external.get("model_id") or "",
        }
        if not backend["base_url"] or not backend["model_id"]:
            return {"error": "external_api.base_url and model_id required"}
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
            return {"error": f"unknown provider {provider_id}"}

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
            try:
                mgr.load(provider_id)
            except Exception as e:  # noqa: BLE001
                return {"error": f"failed to load provider: {e}"}
            backend = {"kind": "provider", "manager": mgr, "provider_id": provider_id}
    else:
        # No provider_id: require the configured 27B GGUF helper — never
        # silently reuse a different Inference/manager model (e.g. a merged
        # project LoRA). Explicit provider_id / external_api remain allowed.
        from finetune_studio.data.prep.generator import (
            helper_resolution_error,
            resolve_helper_backend,
        )

        loaded = resolve_helper_backend()
        if loaded is None:
            return JSONResponse({"error": helper_resolution_error()}, status_code=409)
        backend = dict(loaded)
        log.info(
            "data-prep chat: provider_id omitted; using helper %s backend (%s)",
            backend.get("helper_label"),
            backend.get("kind"),
        )

    # Now check the project directory exists.
    if not project_dir(pid).exists():
        return {"error": f"project {pid} not found"}

    # Run the tool-calling loop.
    full_messages = [{"role": "system", "content": SYSTEM_PROMPT}] + messages
    rounds = 0
    all_tool_calls: list[dict] = []
    last_reply = ""

    for _ in range(max_rounds):
        rounds += 1
        try:
            if backend["kind"] == "external":
                reply_text = await _chat_external(backend, full_messages, gen)
            elif backend["kind"] == "global":
                reply_text = _chat_global_engine(backend, full_messages, gen)
            else:
                reply_text = _chat_local(backend, full_messages, gen)
        except Exception as e:
            log.exception("chat call failed")
            return {"error": f"chat call failed: {e}"}

        tool_calls = _extract_tool_calls(reply_text)
        # Strip Qwen3 chain-of-thought so the visible reply + history
        # message are clean. Falls back to the raw reply_text if the
        # strip is empty.
        visible_reply = _strip_thinking_reply(reply_text) or reply_text
        if _looks_truncated(reply_text):
            cut_msg = _TRUNCATION_MSG.format(n=gen.get("max_tokens", 4096))
            return {
                "ok": False,
                "error": cut_msg,
                "reply": cut_msg,
                "tool_calls": all_tool_calls,
                "rounds": rounds,
                "backend": "external" if backend["kind"] == "external" else "provider",
            }
        if not tool_calls:
            last_reply = visible_reply
            break

        # Execute each tool call, append results to the message history.
        for tc in tool_calls:
            result = _run_tool(pid, tc["name"], tc.get("arguments") or {})
            all_tool_calls.append({
                "name": tc["name"],
                "arguments": tc.get("arguments") or {},
                "result": result,
            })
            full_messages.append({"role": "assistant", "content": visible_reply})
            full_messages.append({
                "role": "user",
                "content": f"TOOL_RESULT {tc['name']}: {json.dumps(result, ensure_ascii=False)}",
            })
        # Loop continues — model sees the tool results and decides what's next.

    return {
        "ok": True,
        "reply": last_reply,
        "tool_calls": all_tool_calls,
        "rounds": rounds,
        "backend": "external" if backend["kind"] == "external" else "provider",
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


def _chat_local(backend: dict, messages: list[dict], gen: dict | None = None) -> str:
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
    except Exception:  # noqa: BLE001
        # Fall back to generate() with a flattened prompt for providers that
        # only support raw text-completion (rare).
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
        except Exception:  # noqa: BLE001
            text = ""
    return text or ""


@router.get("/projects/{pid}/data-prep/chat/tools")
async def list_tools(pid: str):
    """Catalog of tools the chat exposes to the model (useful for debugging
    + for UI to show a help panel)."""
    return {"tools": TOOLS_CATALOG}
