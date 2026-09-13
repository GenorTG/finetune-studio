"""AI judge — evaluates model answers against correct answers.

THE JUDGE'S ONLY JOB: Does the model know the right information?

It does NOT care about:
- Verbosity (a 5-word answer can pass if correct)
- Brevity (a 500-word answer can fail if it's wrong)
- Speaking style (formal, casual, poetic, terse — all fine)
- Word choice or phrasing (paraphrases are fine)
- Length of response (short and long are both fine)

It ONLY checks:
- Are the KEY FACTS present? (entities, names, numbers, relationships)
- Is the knowledge ACCURATE? (no hallucinated facts)
- Is the answer COMPLETE? (covers what the question asks)

If the model knows the right things → pass. If it's missing things → partial. If it's wrong → fail.
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

JUDGE_PROMPT = """You are an expert knowledge evaluator. Your ONLY job is to check whether a model's answer contains the correct information.

You will be given:
- QUESTION: what the model was asked
- CORRECT ANSWER: the ground truth answer (from the training data)
- MODEL ANSWER: what the model under test responded

RULES FOR JUDGING:
1. IGNORE speaking style, tone, length, and verbosity. A model can answer in 5 words or 500 words — both are fine.
2. IGNORE word choice and phrasing. Paraphrases are correct if they mean the same thing.
3. IGNORE whether the answer is formal, casual, poetic, or terse.
4. ONLY check: Does the model answer contain the KEY FACTS? Is the knowledge ACCURATE?

SCORING:
- "pass": The model answer contains all the key facts AND does NOT introduce any incorrect or fabricated information. Even if the model adds extra correct details or is very verbose, it still passes. Even if the answer is extremely brief but hits the key points and adds nothing wrong, it passes.
- "partial": The model answer has some correct facts but is missing important ones, OR includes a minor inaccuracy alongside correct information. A vague or generic answer that hints at the right topic but doesn't give specifics = partial.
- "fail": The model answer contains misinformation, hallucinated facts, or factually incorrect claims. Even if the correct facts are present, adding fabricated details = fail. Completely wrong or irrelevant = fail.

CRITICAL RULE: If the model adds any fact, name, number, or claim that is NOT in the correct answer and is NOT common knowledge (like Paris being a city in France), it must be penalized. Correct facts + made-up facts = partial or fail depending on severity.

DO NOT penalize for:
- Being too short
- Being too long
- Using different words than the correct answer
- Adding extra details (even unrelated ones, as long as the required facts are there)
- Different sentence structure

Example 1:
Q: What is the capital of France?
Correct: Paris
Model: The capital of France is Paris, a city known for the Eiffel Tower.
→ PASS (extra details are fine, key fact "Paris" is there)

Example 2:
Q: What is the capital of France?
Correct: Paris
Model: The capital of France is London.
→ FAIL (wrong fact)

Example 3:
Q: What is the capital of France?
Correct: Paris
Model: A major European city.
→ PARTIAL (vague, doesn't give the specific fact)

Respond in JSON:
{
  "verdict": "pass" | "partial" | "fail",
  "reasoning": "brief explanation focusing on what facts were present/missing/wrong",
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
