"""System + user prompts for Q&A generation, plus the style → hint lookup.

Single responsibility: tell the model how to generate, not parse its output.
"""
from __future__ import annotations

QA_SYSTEM_PROMPT = """You are an expert dataset builder. Given a source passage, you create high-quality
training Q&A pairs that a smaller fine-tuned model could learn from.

CRITICAL OUTPUT RULES:
- Output ONLY a JSON array — nothing else, no preamble, no explanation.
- No markdown fences (no ```json blocks).
- No thinking blocks (no <think>...</think>). Start directly with `[`.
- Each item: {"q": "the question", "a": "the answer"}

Guidelines:
- Questions must be answerable from the passage alone.
- Answers must be concise, factual, and complete.
- Mix factual recall, reasoning, and inference questions.
- Avoid yes/no questions unless the answer is genuinely interesting."""

QA_USER_TEMPLATE = """Source passage:
\"\"\"
{chunk}
\"\"\"

Generate {n} Q&A pairs in the requested style.
Difficulty: {difficulty}
Style: {style_hint}

Respond with ONLY the JSON array. Start with [ and end with ]."""


def style_hint(style: str) -> str:
    """Translate a style name into the prose instruction embedded in the user prompt."""
    return {
        "socratic": "Open with a question that leads the reader to discover the answer.",
        "factual": "Direct, specific questions with precise answers.",
        "analytical": "Why/how questions requiring explanation of mechanisms or reasoning.",
        "comparative": "Compare/contrast with other concepts or approaches.",
        "applied": "Real-world scenarios where the concept is applied or relevant.",
    }.get(style, "Balanced, clear Q&A.")
