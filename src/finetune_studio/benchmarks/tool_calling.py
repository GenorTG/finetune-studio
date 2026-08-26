"""Tool-calling and agentic benchmarks.

WHAT THIS FILE DOES
==================
Tests whether the model can correctly call tools when given an
appropriate prompt. Uses the canonical Jinja renderer from
inference-server to ensure the model's template is respected.

KEY CONCEPTS
============
- Tool call expectations: each test case defines what tools the
  model SHOULD call (and what tools it should NOT call).
- Native rendering: we use the model's actual Jinja template to
  render prompts, which is critical for tool-calling accuracy.
  Without this, the model might output tool calls in the wrong
  format and we'd parse them incorrectly.
- Evaluation: was the right tool called? With the right arguments?
  We use exact match for tool names and string similarity for arguments.
- Why this matters: v21's tool-calling accuracy went from 22% (with
  hardcoded ChatML format) to 90% (with native Jinja rendering).
"""

"""Tool-calling and agentic benchmarks for LLM evaluation."""
import json
import re
from dataclasses import dataclass, field
from typing import Any, cast

# Jinja template support — canonical renderer lives in inference-server.
# We import render_chat (universal Jinja2 renderer with ChatML fallback)
# and extract_template_from_gguf (extracts tokenizer.chat_template from GGUF metadata).
from inference_server.templates.renderer import (
    extract_template_from_gguf,
    render_chat,
)

HAS_INFERENCE_SERVER = True


@dataclass
class ToolCall:
    name: str
    arguments: dict
    raw: str = ""


@dataclass
class AgenticTest:
    name: str
    description: str
    system_prompt: str
    user_message: str
    expected_tools: list  # list of tool names that should be called
    expected_args: dict = field(default_factory=dict)  # expected argument patterns
    forbidden_tools: list = field(default_factory=list)
    category: str = "tool_calling"


class ToolCallEvaluator:
    """Evaluate model's tool-calling capabilities."""

    def parse_tool_call(self, response: str) -> ToolCall | None:
        """Extract tool call from response using multiple formats."""
        # Format 1: <tool_call>{"name": "...", "arguments": {...}}</tool_call>
        match = re.search(r'<tool_call>(.*?)</tool_call>', response, re.DOTALL)
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

        # Format 2: {"tool": "...", "arguments": {...}}
        try:
            data = json.loads(response.strip())
            if "tool" in data or "name" in data:
                return ToolCall(
                    name=data.get("tool", data.get("name", "")),
                    arguments=data.get("arguments", data.get("parameters", {})),
                    raw=response,
                )
        except json.JSONDecodeError:
            pass

        # Format 3: Search for JSON object in response
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

        # Format 4: Qwen/Gemma4 native format - pipe-style tool calls
        # Pattern: ' + repr(p1) + r'call:NAME{args}<tool_call|>
        match = re.search(r'<\|?tool_call\|?>call:(\w+)\{(.+?)\}<\|?tool_call\|?>', response, re.DOTALL)
        if match:
            name = match.group(1)
            args_str = match.group(2).strip()
            # Parse args: KEY:<|"|>VALUE<|"|>  or  KEY:VALUE
            arguments = {}
            for arg_match in re.finditer(r'(\w+):<\|"\|>(.*?)<\|"\|>', args_str, re.DOTALL):
                arguments[arg_match.group(1)] = arg_match.group(2)
            # If no <|"|"> delimited args found, try simpler key:value
            if not arguments:
                for arg_match in re.finditer(r'(\w+):([^,}]+)', args_str):
                    val = arg_match.group(2).strip()
                    if val and not val.startswith('<|'):
                        arguments[arg_match.group(1)] = val
            return ToolCall(name=name, arguments=arguments, raw=match.group(0))

        return None

    def evaluate_tool_call(self, tool_call: ToolCall | None, test: AgenticTest) -> dict:
        """Evaluate a tool call against expected behavior."""
        if tool_call is None:
            # No tool call detected
            # If no tool was expected, this is CORRECT behavior
            if not test.expected_tools:
                return {
                    "correct": True,
                    "tool_called": None,
                    "expected_tools": test.expected_tools,
                    "forbidden_tools": test.forbidden_tools,
                    "args_correct": True,
                    "reason": "No tool needed, none called (correct)",
                }
            return {
                "correct": False,
                "tool_called": None,
                "expected_tools": test.expected_tools,
                "reason": "Expected tool call but none detected",
            }

        # Check if expected tool was called
        correct_tool = tool_call.name in test.expected_tools
        forbidden = tool_call.name in test.forbidden_tools

        # Check arguments
        args_correct = True
        if test.expected_args and correct_tool:
            for key, expected_val in test.expected_args.items():
                if key in tool_call.arguments:
                    actual_val = tool_call.arguments[key]
                    if isinstance(expected_val, list):
                        if not any(v.lower() in str(actual_val).lower() for v in expected_val):
                            args_correct = False
                    elif str(expected_val).lower() not in str(actual_val).lower():
                        args_correct = False

        return {
            "correct": correct_tool and not forbidden and args_correct,
            "tool_called": tool_call.name,
            "arguments": tool_call.arguments,
            "expected_tools": test.expected_tools,
            "forbidden_tools": test.forbidden_tools,
            "args_correct": args_correct,
            "raw": tool_call.raw,
        }


# ============================================================
# STANDARD TOOL-CALLING TEST SUITE
# ============================================================

TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for information",
            "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Calculate a math expression",
            "parameters": {"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "file_read",
            "description": "Read contents of a file",
            "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "note_save",
            "description": "Save a note to memory",
            "parameters": {"type": "object", "properties": {"content": {"type": "string"}, "category": {"type": "string"}}, "required": ["content"]}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "rag_search",
            "description": "Search the knowledge base for relevant information",
            "parameters": {"type": "object", "properties": {"query": {"type": "string"}, "top_k": {"type": "integer"}}, "required": ["query"]}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "weather_check",
            "description": "Check weather for a location",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}, "unit": {"type": "string"}}, "required": ["city"]}
        }
    },
]


# ============================================================
# JINJA TEMPLATE FUNCTIONS
# ============================================================


TOOL_CALL_TESTS = [
    AgenticTest(
        name="web_search_python",
        description="Should call web_search for Python information",
        system_prompt="You are a helpful assistant with access to tools.",
        user_message="Search for information about Python programming language",
        expected_tools=["web_search"],
        expected_args={"query": ["python", "programming"]},
        category="tool_selection",
    ),
    AgenticTest(
        name="calculator_math",
        description="Should call calculator for math",
        system_prompt="You are a helpful assistant with access to tools.",
        user_message="What is 15 * 37 + 42?",
        expected_tools=["calculator"],
        expected_args={"expression": ["15", "37", "42"]},
        category="tool_selection",
    ),
    AgenticTest(
        name="file_read_request",
        description="Should call file_read when asked to read a file",
        system_prompt="You are a helpful assistant with access to tools.",
        user_message="Read the contents of /home/user/document.txt",
        expected_tools=["file_read"],
        expected_args={"path": ["/home/user/document.txt", "document.txt"]},
        category="tool_selection",
    ),
    AgenticTest(
        name="note_save",
        description="Should call note_save when asked to save something",
        system_prompt="You are a helpful assistant with access to tools.",
        user_message="Save a note: Meeting with client at 3pm tomorrow",
        expected_tools=["note_save"],
        expected_args={"content": ["meeting", "client", "3pm"]},
        category="tool_selection",
    ),
    AgenticTest(
        name="rag_search_project",
        description="Should call rag_search for company/project info",
        system_prompt="You are a helpful assistant with access to a company knowledge base.",
        user_message="What projects has the company completed?",
        expected_tools=["rag_search"],
        expected_args={"query": ["project", "company"]},
        category="rag",
    ),
    AgenticTest(
        name="weather_check",
        description="Should call weather_check for weather info",
        system_prompt="You are a helpful assistant with access to tools.",
        user_message="What's the weather in Warsaw?",
        expected_tools=["weather_check"],
        expected_args={"city": ["Warsaw", "warsaw"]},
        category="tool_selection",
    ),
    AgenticTest(
        name="no_tool_needed",
        description="Should NOT call a tool for simple questions",
        system_prompt="You are a helpful assistant with access to tools.",
        user_message="What is 2 + 2?",
        expected_tools=[],
        forbidden_tools=["calculator"],
        category="no_tool",
    ),
    AgenticTest(
        name="greeting_no_tool",
        description="Should NOT call a tool for greetings",
        system_prompt="You are a helpful assistant with access to tools.",
        user_message="Hello! How are you?",
        expected_tools=[],
        forbidden_tools=["web_search", "calculator", "file_read", "note_save", "rag_search", "weather_check"],
        category="no_tool",
    ),
    AgenticTest(
        name="multi_step_search_then_save",
        description="Should call search first, then save note",
        system_prompt="You are a helpful assistant with access to tools.",
        user_message="Search for information about climate change and save a summary note",
        expected_tools=["web_search", "note_save"],
        category="multi_step",
    ),
    AgenticTest(
        name="knowledge_vs_tool",
        description="Should use rag_search for company-specific questions",
        system_prompt="You are an assistant for Acme Corp. You have access to the company knowledge base.",
        user_message="What is our company revenue for Q3?",
        expected_tools=["rag_search"],
        expected_args={"query": ["revenue", "Q3", "company"]},
        category="rag",
    ),
]


def get_tool_system_prompt(tools: list[dict] | None = None) -> str:
    """Get system prompt with tool definitions (generic format)."""
    if tools is None:
        tools = TOOL_DEFINITIONS
    prompt = "You are a helpful assistant with access to tools. When you need to use a tool, respond with ONLY a JSON object:\n"
    prompt += '{"name": "tool_name", "arguments": {"arg": "value"}}\n\n'
    prompt += "Available tools:\n"
    for t in tools:
        # cast: tools is list[dict] but mypy can't resolve nested dict key access on dict[Any]
        # The actual structure is dict like OpenAI function-calling: {type, function: {name, description, parameters: {properties: {...}}}}
        t_dict = cast("dict[str, Any]", t)
        fn = t_dict["function"]
        prompt += f"- {fn['name']}: {fn['description']}\n"
        params = fn.get("parameters", {}).get("properties", {})
        if params:
            prompt += f"  Parameters: {', '.join(params.keys())}\n"
    prompt += "\nIf no tool is needed, respond normally.\n"
    return prompt


# extract_template_from_gguf, render_chat, _render_chatml_fallback are
# imported at the top of this module from inference_server.templates.renderer.
# Do NOT re-define them here — keep a single source of truth.


def build_tool_prompt_for_model(model_path, test, tools=None):
    """Build the full prompt for a test, using the model's native template."""
    if tools is None:
        tools = TOOL_DEFINITIONS
    info = extract_template_from_gguf(model_path)
    messages = [
        {"role": "system", "content": test.system_prompt},
        {"role": "user", "content": test.user_message},
    ]
    return render_chat(
        template_str=info["chat_template"],
        messages=messages,
        tools=tools,
        bos_token=info["bos_token"],
        eos_token=info["eos_token"],
        add_generation_prompt=True,
    )


def detect_tool_format(model_path):
    """Detect the tool call format a model uses by inspecting its template."""
    info = extract_template_from_gguf(model_path)
    template = info["chat_template"]
    if "format_function_declaration" in template or "<|tool>" in template:
        return "gemma4_qwen_style"
    elif "AVAILABLE_TOOLS" in template:
        return "mistral_style"
    elif "<tool_call>" in template:
        return "generic_xml"
    elif info["supports_tools"]:
        return "native_function_calling"
    else:
        return "generic_json"
