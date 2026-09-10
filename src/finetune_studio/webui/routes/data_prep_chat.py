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
from typing import Any, Optional

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

Rules:
1. ALWAYS call `list_sources` first when the user mentions "the files" or "the data" — you need to see what's available before generating anything.
2. Call `read_source` on each file you intend to mine, so the content is fresh in your context.
3. Generate Q&A pairs that test ACTUAL knowledge from the text — not generic questions. The answers should quote or closely paraphrase the source.
4. Aim for 3-8 pairs per source by default. Cover key facts, definitions, cause/effect, comparison, and applied reasoning.
5. Call `create_qa_pairs` with the full batch in one call, not one pair per call. Use a tool call, not plain prose.
6. Be terse in prose — the data does the talking. No filler, no preamble between tool calls.

If you need to call a tool, respond with EXACTLY one tool_call block:
<tool_call>{"name":"tool_name","arguments":{...}}</tool_call>

When you are done and have no more tool calls to make, respond with a single short sentence summarising what you created. Do NOT wrap the summary in a tool_call tag."""

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
                except Exception:
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
            # Trim very long content to keep the context window healthy.
            text = src.get("text") or src.get("parsed_text") or ""
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
    except Exception as e:  # noqa: BLE001
        log.exception("tool %s failed", name)
        return {"error": f"tool {name} failed: {e}"}


def _extract_tool_calls(text: str) -> list[dict]:
    """Pull `<tool_call>{...}</tool_call>` blocks out of a model reply."""
    calls = []
    for m in TOOL_CALL_RE.finditer(text):
        try:
            obj = json.loads(m.group(1))
            name = obj.get("name")
            args = obj.get("arguments") or {}
            if isinstance(name, str) and isinstance(args, dict):
                calls.append({"name": name, "arguments": args})
        except Exception:
            continue
    return calls


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
      provider_id: str      — use the active ModelManager provider (mutually
                              exclusive with external_api)
      external_api: dict    — {base_url, api_key, model_id, name?} for
                              OpenAI-compatible endpoints
      max_rounds: int       — cap tool-call loop iterations (default 6)
      gen: dict             — generation kwargs merged with sensible defaults
                              (temperature, max_tokens, top_p, top_k,
                              repeat_penalty). Used for both backend paths.

    Returns:
      {ok, reply, tool_calls, rounds}
    """
    body = await request.json()
    project_id: str = body.get("project_id") or pid
    messages: list[dict] = body.get("messages") or []
    if not messages:
        return {"error": "messages required"}
    provider_id: Optional[str] = body.get("provider_id")
    external: Optional[dict] = body.get("external_api")
    max_rounds = max(1, min(int(body.get("max_rounds") or MAX_TOOL_ROUNDS), 12))

    # Generation kwargs — allow per-request override. Anything not supplied
    # falls back to sane defaults for tool-calling (low temperature, modest
    # max_tokens). We clamp types here so junk from the frontend doesn't
    # crash llama_cpp.
    gen_raw = body.get("gen") or {}
    gen: dict[str, Any] = {}
    if "temperature" in gen_raw:
        try: gen["temperature"] = max(0.0, min(2.0, float(gen_raw["temperature"])))
        except Exception: pass
    if "max_tokens" in gen_raw:
        try: gen["max_tokens"] = max(32, min(8192, int(gen_raw["max_tokens"])))
        except Exception: pass
    if "top_p" in gen_raw:
        try: gen["top_p"] = max(0.0, min(1.0, float(gen_raw["top_p"])))
        except Exception: pass
    if "top_k" in gen_raw:
        try: gen["top_k"] = max(0, int(gen_raw["top_k"]))
        except Exception: pass
    if "repeat_penalty" in gen_raw:
        try: gen["repeat_penalty"] = max(0.5, min(2.0, float(gen_raw["repeat_penalty"])))
        except Exception: pass
    gen.setdefault("temperature", 0.2)
    gen.setdefault("max_tokens", 1024)

    # Resolve the chat backend (provider OR external API). Validate this
    # before touching the filesystem so a missing project doesn't mask a
    # backend-config error (and vice versa).
    backend: Optional[dict] = None
    if external:
        try:
            import httpx  # noqa: F401
        except Exception:
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
        from finetune_studio.models.manager import get_manager
        mgr = get_manager()
        cfg = mgr.get_provider(provider_id)
        if not cfg:
            return {"error": f"unknown provider {provider_id}"}
        try:
            mgr.load(provider_id)
        except Exception as e:  # noqa: BLE001
            return {"error": f"failed to load provider: {e}"}
        backend = {"kind": "provider", "manager": mgr, "provider_id": provider_id}
    else:
        return {"error": "either provider_id or external_api required"}

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
            else:
                reply_text = _chat_local(backend, full_messages, gen)
        except Exception as e:  # noqa: BLE001
            log.exception("chat call failed")
            return {"error": f"chat call failed: {e}"}

        tool_calls = _extract_tool_calls(reply_text)
        if not tool_calls:
            last_reply = reply_text
            break

        # Execute each tool call, append results to the message history.
        for tc in tool_calls:
            result = _run_tool(pid, tc["name"], tc.get("arguments") or {})
            all_tool_calls.append({
                "name": tc["name"],
                "arguments": tc.get("arguments") or {},
                "result": result,
            })
            full_messages.append({"role": "assistant", "content": reply_text})
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
        "max_tokens": g.get("max_tokens", 1024),
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
            except Exception:
                args = {}
            block = json.dumps({"name": name, "arguments": args})
            return f"<tool_call>{block}</tool_call>"
    return msg.get("content") or ""


def _chat_local(backend: dict, messages: list[dict], gen: dict | None = None) -> str:
    """One round of chat via the local ModelManager provider.

    Renders the structured messages into a single prompt (system prefix +
    role-tagged turns) so the model's text-completion interface can
    answer. The system prompt already tells the model to emit
    <tool_call>...</tool_call> when it wants a tool.
    """
    mgr = backend["manager"]
    sys_prefix, rendered = _messages_to_prompt(messages)
    parts = []
    if sys_prefix:
        parts.append(sys_prefix)
    for m in rendered:
        parts.append(f"{m['role'].upper()}: {m['content']}")
    parts.append("ASSISTANT:")
    prompt = "\n\n".join(parts)
    g = gen or {}
    _temp = g.get("temperature", 0.2)
    _max  = g.get("max_tokens", 1024)
    _topp = g.get("top_p", 0.9)
    try:
        text = mgr.generate(
            prompt,
            max_tokens=_max,
            temperature=_temp,
            top_p=_topp,
        )
    except Exception:
        # Fall back to chat() if the provider supports it.
        text = mgr.chat(rendered, max_tokens=_max, temperature=_temp, top_p=_topp)
    return text or ""


@router.get("/projects/{pid}/data-prep/chat/tools")
async def list_tools(pid: str):
    """Catalog of tools the chat exposes to the model (useful for debugging
    + for UI to show a help panel)."""
    return {"tools": TOOLS_CATALOG}
