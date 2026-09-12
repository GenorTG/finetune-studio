"""AI judge — evaluates model answers against correct answers.

Uses an external model (configurable) to score each transcript:
- pass: model answered correctly and completely
- partial: model answered partially or with minor inaccuracies
- fail: model answered incorrectly or not at all

The judge prompt is designed to be fair: it gives the judge the question,
the correct answer, and the model's answer, then asks for a verdict.
"""

from __future__ import annotations

import json
import os
import time
from typing import Literal

Verdict = Literal["pass", "fail", "partial", ""]

# Default judge model — can be overridden via env or settings
DEFAULT_JUDGE_MODEL = os.environ.get("FTS_JUDGE_MODEL", "gpt-4o-mini")
DEFAULT_JUDGE_API = os.environ.get("FTS_JUDGE_API", "https://api.openai.com/v1")
DEFAULT_JUDGE_KEY = os.environ.get("FTS_JUDGE_API_KEY", "")

JUDGE_PROMPT = """You are an expert evaluator. Your task is to judge whether a model's answer to a question is correct.

You will be given:
- QUESTION: what the model was asked
- CORRECT ANSWER: the ground truth answer (from the training data or test author)
- MODEL ANSWER: what the model under test responded

Score the model answer as:
- "pass": the answer is correct, complete, and accurate
- "partial": the answer is mostly correct but incomplete, slightly inaccurate, or verbose
- "fail": the answer is wrong, irrelevant, or the model refused to answer

Be fair: the model answer doesn't need to match the correct answer word-for-word. It just needs to convey the same correct information. Minor phrasing differences are fine. Missing important details = partial. Hallucinated facts = fail.

Respond in JSON:
{
  "verdict": "pass" | "partial" | "fail",
  "reasoning": "brief explanation of your judgment",
  "confidence": 0.0 to 1.0
}
"""


def build_judge_messages(question: str, correct_answer: str, model_answer: str) -> list[dict]:
    """Build the chat messages for the judge prompt."""
    user_content = (
        f"QUESTION:\n{question}\n\n"
        f"CORRECT ANSWER:\n{correct_answer}\n\n"
        f"MODEL ANSWER:\n{model_answer}"
    )
    return [
        {"role": "system", "content": JUDGE_PROMPT},
        {"role": "user", "content": user_content},
    ]


def judge_case_ai(
    question: str,
    correct_answer: str,
    model_answer: str,
    model: str = DEFAULT_JUDGE_MODEL,
    api_url: str = DEFAULT_JUDGE_API,
    api_key: str = DEFAULT_JUDGE_KEY,
) -> tuple[Verdict, str, float]:
    """Send one case to an AI judge. Returns (verdict, reasoning, confidence)."""
    if not api_key:
        # No API key configured — can't judge with AI
        return "", "no API key configured for AI judge", 0.0

    messages = build_judge_messages(question, correct_answer, model_answer)

    try:
        import urllib.request
        import urllib.error

        payload = json.dumps({
            "model": model,
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": 500,
            "response_format": {"type": "json_object"},
        }).encode()

        req = urllib.request.Request(
            f"{api_url}/chat/completions",
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            method="POST",
        )

        with urllib.request.urlopen(req, timeout=60) as resp:
            result = json.loads(resp.read())

        content = result["choices"][0]["message"]["content"]
        data = json.loads(content)

        verdict = data.get("verdict", "")
        if verdict not in ("pass", "partial", "fail"):
            verdict = "fail"

        return verdict, data.get("reasoning", ""), float(data.get("confidence", 0.5))

    except Exception as e:
        return "", f"AI judge error: {e}", 0.0


def judge_case_local(
    engine,
    question: str,
    correct_answer: str,
    model_answer: str,
    think: bool = False,
) -> tuple[Verdict, str, float]:
    """Judge using a local model (the inference engine)."""
    messages = build_judge_messages(question, correct_answer, model_answer)
    try:
        raw = engine.generate(messages, max_tokens=500, temperature=0.0, think=think)
        # Try to parse JSON from response
        data = json.loads(raw)
        verdict = data.get("verdict", "")
        if verdict not in ("pass", "partial", "fail"):
            verdict = "fail"
        return verdict, data.get("reasoning", ""), float(data.get("confidence", 0.5))
    except json.JSONDecodeError:
        # Fallback: try to extract verdict from plain text
        low = raw.lower()
        if "pass" in low and "partial" not in low:
            return "pass", raw, 0.5
        if "partial" in low:
            return "partial", raw, 0.5
        return "fail", raw, 0.5
    except Exception as e:
        return "", f"local judge error: {e}", 0.0
