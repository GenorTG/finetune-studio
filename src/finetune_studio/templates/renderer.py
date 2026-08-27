"""Jinja2 template rendering engine for model-specific chat formatting.

==================================================================
WHAT THIS FILE DOES (read this first)
==================================================================
This file is the SINGLE SOURCE OF TRUTH for converting a list of chat
messages into the specific text format that a particular LLM model
expects to see as input.

Every LLM (Large Language Model) has its own "chat template" — a
recipe that says:
    "When you see a system message, put <|system|>...</|system|> around it"
    "When you see a user message, put <|user|>...</|user|> around it"
    "When you want the model to start talking, put <|assistant|> at the end"

Different models use different formats. This file:
1. Loads a model's template (a Jinja2 string) from the GGUF file
2. Renders the template with the actual messages
3. Falls back to a generic ChatML format if no template is available

If you only read one file in this codebase, read this one — it's the
foundation of how we talk to LLMs.

==================================================================
KEY CONCEPTS
==================================================================
- GGUF: the file format LLMs are stored in (a quantized binary format)
- Chat template: a Jinja2 string that defines how to format messages
- Jinja2: a Python templating engine (like a fancy "fill in the blanks" tool)
- ChatML: a simple, generic chat format used as our fallback
- BOS/EOS: "Beginning Of Sequence" / "End Of Sequence" special tokens
- Tool calling: when the model answers NOT with text, but with a request
  to call a function (e.g., "call the calculator with 2+2")
"""

# ─── IMPORTS ─────────────────────────────────────────────────────────────
# json: Python's built-in JSON encoder/decoder. We use it to convert
# Python objects (lists, dicts) into strings the template can use.
import json

# `Any` is a type hint that means "any type of variable". Used in
# function signatures to accept any value without forcing a specific type.
from typing import Any

# Jinja2 is a templating engine. It lets us write a template string
# like "Hello {{ name }}" and replace {{ name }} with actual data.
# We use two specific classes from it:
#   - BaseLoader: loads templates from strings (not files)
#   - Environment: the main Jinja2 object that holds configuration
from jinja2 import BaseLoader, Environment

# ─── CONSTANTS ───────────────────────────────────────────────────────────
# CHATML_CLOSE is the token that ends each message in our fallback
# ChatML format. ChatML uses tags like <|user|>/<|im_end|> to mark
# message boundaries. We put the closing token name in a constant
# so the rest of the code can refer to it by name and we can change
# it in one place if needed.
#
# Why is it in a constant? Because some shell heredocs and tooling
# treat specific closing tags like `</parameter>` as special tokens,
# and indirection through a constant avoids those edge cases.
CHATML_CLOSE = "im_end"


# ═════════════════════════════════════════════════════════════════════════
# MAIN FUNCTION: render_chat
# ═════════════════════════════════════════════════════════════════════════
def render_chat(
    template_str: str,
    messages: list[dict[str, Any]],
    tools: list[dict] | None = None,
    bos_token: str = "",
    eos_token: str = "",
    add_generation_prompt: bool = True,
    enable_thinking: bool = False,
) -> str:
    """Render messages using a Jinja2 chat template.

    This is the MAIN function of this file. Given a model's template
    string and a list of messages, it produces the exact text string
    that should be fed to the model.

    Parameters
    ----------
    template_str : str
        A Jinja2 template string from the GGUF file. Examples:
        - Gemma4: "{% for m in messages %}<|turn|>{{ m.role }}\n{{ m.content }}<turn|>{% endfor %}"
        - ChatML: "<|im_start|>{{ role }}\n{{ content }}<|im_end|>"
    messages : list[dict]
        The conversation history. Each dict has "role" (system/user/assistant/tool)
        and "content" (the actual text). Example:
        [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi!"}]
    tools : list[dict] | None
        Optional list of tool definitions the model can call. Format
        follows OpenAI's function calling spec. None means no tools.
    bos_token : str
        The "Beginning Of Sequence" token, prepended to the start.
        Empty string means don't add one.
    eos_token : str
        The "End Of Sequence" token, appended to the end. Empty string
        means don't add one.
    add_generation_prompt : bool
        If True, appends the marker that tells the model "now it's
        your turn to respond" (e.g., "<|assistant|>\n"). Defaults to True.
    enable_thinking : bool
        Some models (like Qwen3) have a "thinking" mode where they
        reason before answering. If True, the template may add a
        `<|thinking|>` block. Defaults to False.

    Returns
    -------
    str
        The fully-rendered prompt ready to send to the model.

    Notes
    -----
    Universal renderer — works with any model's template as long as
    it's a valid Jinja2 string. Handles both simple (ChatML) and
    complex (Gemma4 with tool support) templates.
    """
    # ── Step 1: Check if we have a template to work with ──
    # If template_str is empty (falsy), we use the fallback ChatML
    # renderer. This happens for models that don't store a template
    # in their GGUF metadata.
    if not template_str:
        return _render_chatml_fallback(messages, tools, bos_token, add_generation_prompt)

    # ── Step 2: Try to render with the model's template ──
    try:
        # Create a Jinja2 Environment. An "environment" is the main
        # Jinja2 object that holds configuration and processes templates.
        # We use BaseLoader because our template is a string (not a file).
        # autoescape=False means we don't HTML-escape the output (we want
        # raw model input, not HTML safety).
        env = Environment(loader=BaseLoader(), autoescape=False)

        # Add a custom filter called "tojson" that converts Python
        # objects to JSON strings. Templates can use it like:
        #   {{ my_dict | tojson }}
        # This is how templates handle tool/function definitions.
        env.filters["tojson"] = lambda x: json.dumps(x, ensure_ascii=False)

        # Parse the template string into a Jinja2 Template object.
        # This compiles the template so it can be rendered quickly.
        template = env.from_string(template_str)

        # Actually render the template. We pass in all the variables
        # the template might need. Most templates use `messages` and
        # `add_generation_prompt`; some templates use multiple names
        # for the same thing (functions, tools, tool_definitions) so
        # we pass all of them — the template will pick what it needs.
        return template.render(
            messages=messages,
            tools=tools or [],          # Empty list if no tools
            bos_token=bos_token,
            eos_token=eos_token,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=enable_thinking,
            functions=tools or [],      # Alias for "tools" (some templates use this)
            tool_definitions=tools or [],  # Alias for "tools" (some templates use this)
        )
    # ── Step 3: If rendering fails, fall back to ChatML ──
    # Why? Templates can have syntax errors (if the GGUF is malformed).
    # We don't want a broken template to crash the entire inference server.
    # Logging the error and falling back is more graceful than crashing.
    except Exception as e:  # noqa: BLE001
        print(f"Template render error: {e}, falling back to ChatML")
        return _render_chatml_fallback(messages, tools, bos_token, add_generation_prompt)


# ─── HELPER: _chatml_wrap ────────────────────────────────────────────────
def _chatml_wrap(tag, content_str):
    """Wrap content in ChatML tags.

    Helper function for the fallback renderer. Takes a tag name
    (like "user" or "system") and content, and wraps it like:
        <|user|>content</im_end|>

    The leading underscore in the name signals "this is internal —
    don't call me from outside the file" (Python convention).

    Parameters
    ----------
    tag : str
        The tag name (e.g., "user", "system", "assistant")
    content_str : str
        The text to wrap

    Returns
    -------
    str
        The wrapped text with ChatML tags
    """
    # Build the opening tag like <|user|>
    open_tag = f"<|{tag}|>"
    # Build the closing tag like </im_end|>
    # We use CHATML_CLOSE constant instead of writing "</im_end|>" directly
    close_tag = f"</{CHATML_CLOSE}>"
    # Concatenate open + content + close
    return f"{open_tag}{content_str}{close_tag}"


# ═════════════════════════════════════════════════════════════════════════
# FALLBACK RENDERER: _render_chatml_fallback
# ═════════════════════════════════════════════════════════════════════════
def _render_chatml_fallback(
    messages: list[dict[str, Any]],
    tools: list[dict] | None = None,
    bos_token: str = "",
    add_generation_prompt: bool = True,
) -> str:
    """ChatML fallback for models without templates.

    If a model doesn't have a Jinja2 template in its GGUF file, we use
    this generic ChatML format. It's the "lowest common denominator"
    that works with most open-source models.

    Parameters
    ----------
    messages : list[dict]
        The conversation history.
    tools : list[dict] | None
        Optional tool definitions (unused in fallback, but accepted
        for API compatibility).
    bos_token : str
        Beginning-of-sequence token to prepend.
    add_generation_prompt : bool
        If True, adds "<|assistant|>\n" at the end so the model
        knows to start its response.

    Returns
    -------
    str
        The rendered ChatML prompt.
    """
    # Start with the BOS token if provided. List comprehension-style:
    # if bos_token is truthy, return [bos_token], else return [].
    # This means we'll either have an empty list or a one-element list
    # containing the BOS token.
    parts = [bos_token] if bos_token else []

    # ── Loop through each message and wrap it in role-specific tags ──
    for msg in messages:
        # Extract the role (system/user/assistant/tool) and content
        role = msg["role"]
        content = msg.get("content", "")  # Default to empty string if no content

        # Dispatch based on role. We use if/elif instead of a dict
        # lookup because the formatting is slightly different for each.
        if role == "system":
            # System messages: instructions to the model (e.g., "You are a helpful assistant")
            parts.append(_chatml_wrap("system", f"\n{content}"))
        elif role == "user":
            # User messages: the human's input
            parts.append(_chatml_wrap("user", f"\n{content}"))
        elif role == "assistant":
            # Assistant messages: the model's previous responses
            parts.append(_chatml_wrap("assistant", f"\n{content}"))
        elif role == "tool":
            # Tool messages: results from a tool call (e.g., calculator returned "4")
            parts.append(_chatml_wrap("tool", f"\n{content}"))
        # If the role is something else, we silently skip it.

    # ── Add the "generation prompt" ──
    # This is the marker that says "now it's your turn to talk".
    # For ChatML, it's just "<|assistant|>\n" — the model sees this
    # and knows to start generating its response.
    if add_generation_prompt:
        parts.append("<|assistant|>\n")

    # Join all parts into a single string. "".join() concatenates
    # every element of the list with no separator between them.
    return "".join(parts)


# ═════════════════════════════════════════════════════════════════════════
# GGUF METADATA READER: extract_template_from_gguf
# ═════════════════════════════════════════════════════════════════════════
def extract_template_from_gguf(model_path: str) -> dict[str, Any]:
    """Extract chat template and tokens from a GGUF file.

    GGUF files (the format LLMs are stored in) embed metadata as
    key-value pairs. One of those keys is "tokenizer.chat_template"
    which contains the Jinja2 template string. This function reads
    the GGUF metadata without loading the entire model into memory.

    Parameters
    ----------
    model_path : str
        Path to the GGUF file (e.g., "/models/chris-ai-v21.Q4_K_M.gguf")

    Returns
    -------
    dict
        A dictionary with these keys:
        - chat_template: str — the Jinja2 template string
        - bos_token: str — beginning-of-sequence token
        - eos_token: str — end-of-sequence token
        - add_bos: bool — whether to prepend BOS by default
        - supports_tools: bool — whether the template has tool support

    Notes
    -----
    We load the model with a tiny context (n_ctx=512) and 1 thread
    just to access the metadata. We don't actually generate anything.
    This is a fast way to read GGUF metadata.
    """
    try:
        # Lazy import: only load llama_cpp when actually needed.
        # This keeps the import time fast for code paths that don't use GGUF.
        from llama_cpp import Llama

        # Load the model with minimal settings. We just need the metadata,
        # so small context and 1 thread is enough. verbose=False suppresses
        # llama.cpp's logging output.
        llm = Llama(model_path=model_path, n_ctx=512, n_threads=1, verbose=False)

        # Get the full metadata dict. GGUF metadata is a flat dict of
        # string keys to various value types (int, float, str, bool).
        meta = llm.metadata

        # Extract the template string. The key "tokenizer.chat_template"
        # is the standard GGUF key for Jinja2 chat templates.
        template_str = meta.get("tokenizer.chat_template", "")

        # Extract BOS/EOS tokens with reasonable defaults.
        # These are the special tokens that mark the start/end of a sequence.
        bos = meta.get("tokenizer.ggml.bos_token", "<bos>")
        eos = meta.get("tokenizer.ggml.eos_token", "<eos>")

        # add_bos is a boolean flag from GGUF — it's True if the model
        # expects a BOS token at the start of every input.
        add_bos = meta.get("tokenizer.ggml.add_bos_token", True)

        # ── Detect tool support by scanning the template ──
        # We can't just rely on a single flag because GGUF doesn't
        # have a standard "supports_tools" key. Instead, we look for
        # tool-related patterns in the template string.
        tool_indicators = [
            "format_function_declaration",  # Gemma4 uses this macro
            "tool_call",                    # Common in many templates
            "AVAILABLE_TOOLS",              # Some custom templates
            "tool_response",                 # Response block marker
            "<\u200Btool_call>",                  # Qwen-style tool call marker
        ]
        # If ANY of these patterns appear in the template, we assume
        # the model supports tool calling. `any()` returns True if at
        # least one element of the iterable is truthy.
        supports_tools = any(ind in template_str for ind in tool_indicators)

        # Return all the extracted info as a dict. str()/bool() casts
        # ensure the values are JSON-serializable (in case the caller
        # wants to JSON-encode this dict).
        return {
            "chat_template": template_str,
            "bos_token": str(bos),
            "eos_token": str(eos),
            "add_bos": bool(add_bos),
            "supports_tools": supports_tools,
        }
    # If anything goes wrong (file not found, corrupt GGUF, etc.),
    # we return sensible defaults instead of crashing.
    except Exception as e:  # noqa: BLE001
        print(f"Failed to extract template from {model_path}: {e}")
        return {
            "chat_template": "",
            "bos_token": "<bos>",
            "eos_token": "<eos>",
            "add_bos": True,
            "supports_tools": False,
        }


# ═════════════════════════════════════════════════════════════════════════
# CONVENIENCE FUNCTION: render_with_model_template
# ═════════════════════════════════════════════════════════════════════════
def render_with_model_template(
    model_path: str,
    messages: list[dict[str, Any]],
    tools: list[dict] | None = None,
    add_generation_prompt: bool = True,
) -> str:
    """One-shot: extract template from GGUF and render messages.

    This is a convenience function that combines two steps:
      1. Extract the template from the GGUF file
      2. Render the messages with that template

    Use this when you have a model path and want to render messages
    without manually calling extract_template_from_gguf() first.

    Parameters
    ----------
    model_path : str
        Path to the GGUF file.
    messages : list[dict]
        The conversation history.
    tools : list[dict] | None
        Optional tool definitions.
    add_generation_prompt : bool
        Whether to add the generation prompt at the end.

    Returns
    -------
    str
        The rendered prompt string.
    """
    # Step 1: Extract the template info from the GGUF
    tmpl = extract_template_from_gguf(model_path)

    # Step 2: Render the messages using the extracted template
    # We pass the template string and all the tokens extracted in step 1.
    return render_chat(
        template_str=tmpl["chat_template"],
        messages=messages,
        tools=tools,
        bos_token=tmpl["bos_token"],
        eos_token=tmpl["eos_token"],
        add_generation_prompt=add_generation_prompt,
    )
