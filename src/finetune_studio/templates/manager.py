"""Template manager — caches templates per model name.

WHAT THIS FILE DOES
==================
The renderer.py file renders messages. The manager.py file provides
a higher-level interface: "render messages for model X" — looking up
the template from cache or loading it if not cached.

KEY CONCEPTS
============
- Caching: loading a GGUF and reading its template is slow (reads the
  whole file). We cache templates by model path so we only do it once.
- ChatTemplate: a dataclass holding the template string and tokens.
- Model name resolution: callers can use "chris-ai-v21" instead of
  the full path; we look up the path in a registry.
"""

"""Chat template manager - extracts templates from GGUF and handles tool calling."""
import json
from dataclasses import dataclass
from typing import Any

from .renderer import extract_template_from_gguf, render_chat


@dataclass
class ChatTemplate:
    name: str = "unknown"
    template: str = ""
    tool_format: str = "generic"
    supports_tools: bool = False
    custom_system_prompt: str = ""
    bos_token: str = ""
    eos_token: str = ""
    add_bos: bool = True


class TemplateManager:
    """Manages chat templates per model. Extracts from GGUF, caches, renders."""

    def __init__(self):
        self.templates: dict[str, ChatTemplate] = {}

    def register_from_gguf(self, model_path: str, model_name: str | None = None) -> ChatTemplate:
        """Extract and register template from a GGUF file."""
        name = model_name or model_path.split("/")[-1].replace(".gguf", "")
        info = extract_template_from_gguf(model_path)

        # Detect format from template
        fmt = _detect_format(info["chat_template"])

        tmpl = ChatTemplate(
            name=name,
            template=info["chat_template"],
            tool_format=fmt,
            supports_tools=info["supports_tools"],
            bos_token=info["bos_token"],
            eos_token=info["eos_token"],
            add_bos=info["add_bos"],
        )
        self.templates[name] = tmpl
        return tmpl

    def get_template(self, model_name: str) -> ChatTemplate:
        return self.templates.get(model_name, ChatTemplate())

    def render(
        self,
        model_name: str,
        messages: list[dict[str, Any]],
        tools: list[dict] | None = None,
        add_generation_prompt: bool = True,
    ) -> str:
        """Render messages using the registered template for a model."""
        tmpl = self.get_template(model_name)
        return render_chat(
            template_str=tmpl.template,
            messages=messages,
            tools=tools,
            bos_token=tmpl.bos_token,
            eos_token=tmpl.eos_token,
            add_generation_prompt=add_generation_prompt,
        )

    def build_tool_system_prompt(self, model_name: str | None = None, tools: list | None = None) -> str:
        """Build system prompt with tool definitions."""
        if tools is None:
            tools = DEFAULT_TOOLS

        fmt = "generic"
        if model_name and model_name in self.templates:
            fmt = self.templates[model_name].tool_format

        if fmt in ("qwen", "hermes"):
            return _build_qwen_tool_prompt(tools)
        elif fmt == "gemma4":
            return _build_gemma4_tool_prompt(tools)
        elif fmt == "mistral":
            return _build_mistral_tool_prompt(tools)
        else:
            return _build_generic_tool_prompt(tools)


# ══════════════════════════════════════════════════════════════
# FORMAT DETECTION
# ══════════════════════════════════════════════════════════════

FORMAT_PATTERNS = {
    "qwen": ["<im_start>", "<tool_call>", "<|tool_call|>", "format_function_declaration"],
    "hermes": ["<|system|>", "<|user|>", "<|assistant|>", "<tool_call>"],
    "llama3": ["<|start_header_id|>", "<|end_header_id|>"],
    "mistral": ["[INST]", "[/INST]", "[AVAILABLE_TOOLS]", "[/AVAILABLE_TOOLS]"],
    "gemma": ["<start_of_turn>", "<end_of_turn>"],
    "gemma4": ["<|turn>", "<|tool_response>", "format_function_declaration"],
    "chatml": ["<im_start>", "<im_end>"],
    "phi3": ["<|system|>", "<|user|>", "<|end|>"],
}


def _detect_format(template_str: str) -> str:
    if not template_str:
        return "generic"
    for fmt, markers in FORMAT_PATTERNS.items():
        if any(m in template_str for m in markers):
            return fmt
    return "generic"


# ══════════════════════════════════════════════════════════════
# TOOL PROMPT BUILDERS (per format)
# ══════════════════════════════════════════════════════════════

DEFAULT_TOOLS = [
    {"type": "function", "function": {"name": "web_search", "description": "Search the web", "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
    {"type": "function", "function": {"name": "calculator", "description": "Calculate math", "parameters": {"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]}}},
    {"type": "function", "function": {"name": "file_read", "description": "Read a file", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}},
    {"type": "function", "function": {"name": "note_save", "description": "Save a note", "parameters": {"type": "object", "properties": {"content": {"type": "string"}, "category": {"type": "string"}}, "required": ["content"]}}},
    {"type": "function", "function": {"name": "rag_search", "description": "Search knowledge base", "parameters": {"type": "object", "properties": {"query": {"type": "string"}, "top_k": {"type": "integer"}}, "required": ["query"]}}},
    {"type": "function", "function": {"name": "weather_check", "description": "Check weather", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}},
]


def _build_generic_tool_prompt(tools: list) -> str:
    prompt = 'You have access to these tools. When you need to use one, respond with ONLY a JSON object:\n'
    prompt += '{"name": "tool_name", "arguments": {"arg": "value"}}\n\n'
    prompt += "Available tools:\n"
    for t in tools:
        fn = t["function"]
        prompt += f"- {fn['name']}: {fn['description']}\n"
    prompt += "\nIf no tool is needed, respond normally.\n"
    return prompt


def _build_qwen_tool_prompt(tools: list) -> str:
    return _build_generic_tool_prompt(tools)


def _build_gemma4_tool_prompt(tools: list) -> str:
    prompt = "You have access to these tools. When you need to use a tool, call it using the provided format.\n\n"
    prompt += "Available tools:\n"
    for t in tools:
        fn = t["function"]
        params = fn.get("parameters", {}).get("properties", {})
        param_str = ", ".join(params.keys()) if params else "none"
        prompt += f"- {fn['name']}: {fn['description']} (params: {param_str})\n"
    prompt += "\nIf no tool is needed, respond normally.\n"
    return prompt


def _build_mistral_tool_prompt(tools: list) -> str:
    prompt = "[AVAILABLE_TOOLS]\n"
    for t in tools:
        prompt += json.dumps(t, ensure_ascii=False) + "\n"
    prompt += "[/AVAILABLE_TOOLS]\n"
    return prompt
