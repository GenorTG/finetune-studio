# mypy: disable-error-code="arg-type,call-arg"
"""Agentic Tools playground — routes.

WHAT THIS FILE DOES
==================
Provides an agentic loop playground where the user can:
  - Load a model
  - Enable specific tools (calculator, web_search, RAG lookup)
  - Send a prompt and watch the model reason, call tools, and respond
  - See the full tool call trace

KEY CONCEPTS
============
- Agentic loop: generate → parse tool call → execute tool → append
  result → generate again (up to max_steps).
- Tool call parsing: multiple formats (JSON, <tool_call>, pipe-style)
  extracted from the existing benchmark evaluator.
- Tool execution: calculator (eval), web_search (mock or API),
  rag_search (VectorStore lookup).
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any

from fastapi import APIRouter, Request

router = APIRouter()

# ── Tool call parsing (from benchmarks/tool_calling.py, simplified) ──────


@dataclass
class ToolCall:
    name: str
    arguments: dict[str, Any]
    raw: str = ""


def parse_tool_call(response: str) -> ToolCall | None:
    """Extract tool call from model response using multiple formats."""
    # Format 1: <tool_call>{"name": "...", "arguments": {...}}</tool_call>
    match = re.search(r"<tool_call>(.*?)</tool_call>", response, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(1).strip())
            return ToolCall(
                name=data.get("name", data.get("tool", "")),
                arguments=data.get("arguments", data.get("parameters", {})),
                raw=match.group(0),
            )
        except json.JSONDecodeError:
            pass

    # Format 2: {"tool": "...", "arguments": {...}} or {"name": "...", ...}
    try:
        data = json.loads(response.strip())
        if isinstance(data, dict) and ("tool" in data or "name" in data):
            return ToolCall(
                name=data.get("tool", data.get("name", "")),
                arguments=data.get("arguments", data.get("parameters", {})),
                raw=response,
            )
    except json.JSONDecodeError:
        pass

    # Format 3: JSON object embedded in text
    match = re.search(r'\{[^{}]*"name"[^{}]*\}', response)
    if match:
        try:
            data = json.loads(match.group(0))
            return ToolCall(
                name=data.get("name", data.get("tool", "")),
                arguments=data.get("arguments", {}),
                raw=match.group(0),
            )
        except json.JSONDecodeError:
            pass

    # Format 4: Qwen/Gemma4 pipe-style  <|tool_call|>call:NAME{args}<|tool_call|>
    match = re.search(
        r"<\|?tool_call\|?>call:(\w+)\{(.+?)\}<\|?tool_call\|?>", response, re.DOTALL
    )
    if match:
        name = match.group(1)
        args_str = match.group(2).strip()
        arguments: dict[str, str] = {}
        for m in re.finditer(r'(\w+):<\|"\|>(.*?)<\|"\|>', args_str, re.DOTALL):
            arguments[m.group(1)] = m.group(2)
        if not arguments:
            for m in re.finditer(r"(\w+):([^,}]+)", args_str):
                val = m.group(2).strip()
                if val and not val.startswith("<|"):
                    arguments[m.group(1)] = val
        return ToolCall(name=name, arguments=arguments, raw=match.group(0))

    # Format 5: XML-style <tool_call><name>calc</name><arguments>...</arguments></tool_call>
    match = re.search(
        r"<tool_call>\s*<name>(\w+)</name>\s*<arguments>(.*?)</arguments>\s*</tool_call>",
        response,
        re.DOTALL,
    )
    if match:
        try:
            args = json.loads(match.group(2).strip())
        except json.JSONDecodeError:
            args = {}
        return ToolCall(name=match.group(1), arguments=args, raw=match.group(0))

    return None


# ── Tool execution ───────────────────────────────────────────────────────


def _exec_calculator(args: dict) -> str:
    """Evaluate a math expression safely."""
    expr = args.get("expression", args.get("query", ""))
    if not expr:
        return "Error: no expression provided"
    # Only allow safe math characters
    if not re.fullmatch(r"[\d\s\+\-\*/\.\(\)%^]+", expr):
        return f"Error: unsafe expression '{expr}'"
    try:
        # Replace ^ with ** for Python exponentiation
        py_expr = expr.replace("^", "**")
        result = eval(py_expr, {"__builtins__": {}}, {})  # noqa: S307
        return str(result)
    except Exception as e:
        return f"Error: {e}"


def _exec_web_search(args: dict) -> str:
    """Web search — returns mock if no API key."""
    query = args.get("query", "")
    api_key = os.environ.get("SERPAPI_KEY") or os.environ.get("BRAVE_API_KEY")
    if not api_key:
        return f"[web_search] No search API key configured. Query was: '{query}'"
    # TODO: integrate real search when API key available
    return f"[web_search] Search API key found but integration not yet implemented. Query: '{query}'"


def _exec_rag_search(args: dict, project_id: str | None = None) -> str:
    """RAG search against a project's vector store."""
    rid = args.get("rag_id", "")
    query = args.get("query", "")
    top_k = int(args.get("top_k", 3))

    if not rid:
        return "Error: no rag_id specified"
    if not project_id:
        return "Error: no project context"

    from finetune_studio import db
    from finetune_studio.rag.store import VectorStore

    rag = db.get_rag(rid)
    if not rag:
        return f"Error: RAG '{rid}' not found"

    try:
        store = VectorStore(store_path=rag["store_path"])
        results = store.search(query, top_k=top_k)
        if not results:
            return f"No results found for '{query}' in knowledge base '{rag['name']}'"
        parts = []
        for i, r in enumerate(results, 1):
            parts.append(f"[{i}] (score={r.score:.3f}) {r.text[:300]}")
        return f"Results from '{rag['name']}':\n\n" + "\n\n".join(parts)
    except Exception as e:
        return f"RAG search error: {e}"


def execute_tool(name: str, arguments: dict, project_id: str | None = None) -> str:
    """Execute a tool by name and return result string."""
    if name == "calculator":
        return _exec_calculator(arguments)
    if name == "web_search":
        return _exec_web_search(arguments)
    if name.startswith("rag_"):
        return _exec_rag_search(arguments, project_id)
    return f"Unknown tool: '{name}'"


# ── System prompt builder ────────────────────────────────────────────────


def build_tool_system_prompt(enabled_tools: list[str], project_rags: list[dict]) -> str:
    """Build system prompt section describing available tools."""
    lines = [
        "You are a helpful assistant with access to tools.",
        "When you need to use a tool, respond with ONLY a JSON object:",
        '{"name": "tool_name", "arguments": {"arg": "value"}}',
        "",
        "Available tools:",
    ]

    tool_descriptions = {
        "calculator": "  - calculator: Evaluate a math expression. Arguments: {expression: string}",
        "web_search": "  - web_search: Search the web for information. Arguments: {query: string}",
    }

    for tool in enabled_tools:
        if tool in tool_descriptions:
            lines.append(tool_descriptions[tool])
        elif tool.startswith("rag_"):
            rid = tool[4:]
            rag = next((r for r in project_rags if r["id"] == rid), None)
            desc = rag["description"] if rag else "knowledge base search"
            lines.append(
                f"  - {tool}: Search '{desc}'. Arguments: {{query: string, top_k: integer}}"
            )

    lines.append("")
    lines.append("If no tool is needed, respond normally with your answer.")
    return "\n".join(lines)


# ── API routes ───────────────────────────────────────────────────────────


@router.get("/status")
async def status():
    """Return the currently loaded model info."""
    from finetune_studio.webui.app import inference_engine

    return {
        "loaded": inference_engine.model is not None,
        "model_path": inference_engine.model_path,
        "is_gguf": inference_engine.is_gguf,
    }


@router.get("/load")
async def load_model(path: str = ""):
    """Load a model into the shared inference engine."""
    from finetune_studio.webui.app import inference_engine

    if not path:
        return {"error": "No model path provided"}
    try:
        inference_engine.load(path)
        return {"status": "loaded", "model": path}
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)}


@router.get("/projects/{pid}/tools")
async def list_tools(pid: str):
    """List available tools for a project."""
    from finetune_studio import db

    rags = db.list_rags(pid)
    tools = [
        {"name": "calculator", "description": "Evaluate math expressions", "enabled_by_default": True},
        {"name": "web_search", "description": "Search the web for information", "enabled_by_default": False},
    ]
    for rag in rags:
        tools.append({
            "name": f"rag_{rag['id']}",
            "description": rag["description"] or rag["name"],
            "enabled_by_default": True,
        })
    return tools


@router.post("/projects/{pid}/run")
async def agentic_run(pid: str, request: Request):
    """Run the agentic loop: model → tool call → tool result → repeat."""
    from finetune_studio import db
    from finetune_studio.webui.app import inference_engine

    body = await request.json()
    messages = body.get("messages", [])
    enabled_tools = body.get("enabled_tools", [])
    model_path = body.get("model_path", "")
    max_tokens = body.get("max_tokens", 512)
    temperature = body.get("temperature", 0.7)
    max_steps = min(body.get("max_steps", 6), 10)

    # Ensure model is loaded
    if inference_engine.model is None:
        if model_path:
            try:
                inference_engine.load(model_path)
            except Exception as e:  # noqa: BLE001
                return {"error": f"Failed to load model: {e}", "steps": [], "final_response": ""}
        else:
            return {"error": "No model loaded. Load a model first.", "steps": [], "final_response": ""}

    # Build tool-augmented system prompt
    project_rags = db.list_rags(pid)
    tool_prompt = build_tool_system_prompt(enabled_tools, project_rags)

    # Prepend or augment system message
    full_messages = []
    has_system = False
    for msg in messages:
        if msg["role"] == "system":
            full_messages.append({"role": "system", "content": msg["content"] + "\n\n" + tool_prompt})
            has_system = True
        else:
            full_messages.append(msg)
    if not has_system:
        full_messages.insert(0, {"role": "system", "content": tool_prompt})

    # Agentic loop
    steps: list[dict[str, Any]] = []
    sources: list[dict] = []
    final_response = ""

    for step_num in range(max_steps):
        try:
            response_text = inference_engine.generate(
                full_messages, max_tokens=max_tokens, temperature=temperature
            )
        except Exception as e:  # noqa: BLE001
            steps.append({"role": "assistant", "content": f"[Error: {e}]", "tool_calls": [], "tool_results": []})
            final_response = f"[Error during generation: {e}]"
            break

        tool_call = parse_tool_call(response_text)

        if tool_call is None or tool_call.name not in enabled_tools:
            # No tool call — this is the final response
            steps.append({"role": "assistant", "content": response_text, "tool_calls": [], "tool_results": []})
            final_response = response_text
            break

        # Execute the tool
        tool_result = execute_tool(tool_call.name, tool_call.arguments, pid)

        # Record step
        steps.append({
            "role": "assistant",
            "content": response_text,
            "tool_calls": [{"name": tool_call.name, "arguments": tool_call.arguments}],
            "tool_results": [{"name": tool_call.name, "result": tool_result}],
        })

        # Append tool result to messages for next iteration
        full_messages.append({"role": "assistant", "content": response_text})
        full_messages.append({
            "role": "user",
            "content": f"Tool result for {tool_call.name}: {tool_result}\n\nNow continue with your answer.",
        })

        # Track RAG sources
        if tool_call.name.startswith("rag_") and "Results from" in tool_result:
            sources.append({"tool": tool_call.name, "result_preview": tool_result[:500]})
    else:
        # Hit max steps — generate final response
        try:
            response_text = inference_engine.generate(
                full_messages, max_tokens=max_tokens, temperature=temperature
            )
        except Exception as e:
            response_text = f"[Error: {e}]"
        steps.append({"role": "assistant", "content": response_text, "tool_calls": [], "tool_results": []})
        final_response = response_text

    return {"steps": steps, "final_response": final_response, "sources": sources}
