"""Auto-generate benchmark suites from training data.

WHAT THIS FILE DOES
==================
After training, converts the training data (JSONL with Q&A pairs) into
a benchmark suite that tests exactly what the model was trained on.

WHY
===
If you train on Aethermere lore, the benchmark should test Aethermere lore —
not general AI knowledge. This module bridges that gap.

WHAT MAKES IT REALISTIC
========================
- Short answers ("147 years old", "his shadow") are VALID — they test factual recall
- Long explanations test comprehension and expression
- The judge adapts: short answers need exact match, long answers need semantic similarity
- All valid Q&A pairs are included, not just the "nice" ones

FLOW
======
1. Training completes → training data is a JSONL file with conversations
2. Each Q&A pair becomes a BenchmarkCase (question + correct answer + judging hints)
3. Suite is saved as JSON alongside the training output
4. DB records the suite path so the UI can find it
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

from finetune_studio.testing.suite import BenchmarkCase

_USER_ROLES = {"human", "user"}
_ASSISTANT_ROLES = {"gpt", "assistant"}


def _system_turn(conversations: object) -> str:
    """The row's opening system turn (grounded rows: the retrieved CONTEXT), else empty."""
    if not isinstance(conversations, list) or not conversations:
        return ""
    first = conversations[0]
    if not isinstance(first, dict) or str(first.get("from") or first.get("role") or "").lower() != "system":
        return ""
    return str(first.get("value") or first.get("content") or "").strip()


def _first_exchange(conversations: object) -> tuple[str, str]:
    """First user turn and the first assistant turn after it, by role.

    Context-grounded rows open with a system turn that carries the retrieved
    CONTEXT, so positional ``[0]``/``[1]`` would quiz the model with the system
    prompt and expect the real question as the answer.
    """
    if not isinstance(conversations, list):
        return "", ""
    question = ""
    for turn in conversations:
        if not isinstance(turn, dict):
            continue
        role = str(turn.get("from") or turn.get("role") or "").lower()
        text = str(turn.get("value") or turn.get("content") or "").strip()
        if not question:
            if role in _USER_ROLES:
                question = text
        elif role in _ASSISTANT_ROLES:
            return question, text
    return "", ""


def generate_suite_from_training_data(
    data_path: str,
    output_dir: str,
    suite_name: str | None = None,
    max_cases: int | None = None,
    sample_seed: int = 42,
) -> dict:
    """Convert a training JSONL file into a benchmark suite.

    Coverage rule (Genor, 2026-09-20): the suite must test **all** dataset
    rows — 100 rows → 100 cases, 1M rows → 1M cases. ``max_cases`` is an
    explicit opt-in to sampling for huge corpora only; when it kicks in the
    suite is named ``<stem>-sampledKofN`` and the result carries
    ``coverage="sampled"`` so no verdict can be misread as full coverage.

    Args:
        data_path: Path to JSONL training data (conversations format)
        output_dir: Where to save the suite JSON
        suite_name: Name for the suite (defaults to filename stem)
        max_cases: None (default) = full coverage; int = deterministic
            uniform sample of that many rows (seeded, reproducible)
        sample_seed: RNG seed for sampling (default 42)

    Returns:
        {suite_path, case_count, skipped, categories, difficulty,
         coverage, dataset_count, ...}
    """
    data_path = str(data_path)
    output_dir = str(output_dir)

    if not os.path.isfile(data_path):
        return {"error": f"training data not found: {data_path}"}

    # Load training data
    examples = []
    invalid_lines = 0
    with open(data_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                examples.append(json.loads(line))
            except json.JSONDecodeError:
                invalid_lines += 1

    if not examples:
        return {"error": "no valid examples in training data"}

    # Convert to benchmark cases — parse ALL rows first, sample after, so a
    # sampled suite covers the whole file uniformly instead of its head.
    cases = []
    skipped = 0
    categories: dict[str, int] = {}
    difficulty: dict[str, int] = {}
    seen_names: dict[str, int] = {}

    for i, ex in enumerate(examples):
        # SFT suites use conversations; a DPO suite asks the same prompt and
        # expects the chosen response (never the rejected response).
        preference_prompt = ex.get("prompt")
        preference_chosen = ex.get("chosen")
        if preference_prompt is not None and preference_chosen is not None:
            prompt_messages = _preference_messages(preference_prompt, "user")
            chosen_messages = _preference_messages(preference_chosen, "assistant")
            question = next((
                str(m.get("content") or "").strip()
                for m in reversed(prompt_messages)
                if str(m.get("role") or "").lower() in _USER_ROLES
                and str(m.get("content") or "").strip()
            ), "")
            answer = next((
                str(m.get("content") or "").strip()
                for m in reversed(chosen_messages)
                if str(m.get("role") or "").lower() in _ASSISTANT_ROLES
                and str(m.get("content") or "").strip()
            ), "")
            system_prompt = next((
                str(m.get("content") or "").strip()
                for m in prompt_messages
                if str(m.get("role") or "").lower() == "system"
            ), "")
        else:
            question, answer = _first_exchange(ex.get("conversations", []))
            system_prompt = _system_turn(ex.get("conversations", []))

        if not question or not answer:
            skipped += 1
            continue

        # Categorize by question type
        category = _categorize(question, answer)

        # Determine difficulty and judging strategy based on answer characteristics
        _diff, judge_hint = _analyze_difficulty(question, answer)

        # Create a stable name from the question, deduplicated — full-coverage
        # suites on big datasets WILL have colliding question prefixes.
        base = _slugify(question[:60]) or f"case_{i}"
        n = seen_names.get(base, 0)
        seen_names[base] = n + 1
        name = base if n == 0 else f"{base}_{n + 1}"

        cases.append(BenchmarkCase(
            name=name,
            question=question,
            correct_answer=answer,
            category=category,
            context=judge_hint,
            source_id=str(ex.get("source_id") or ""),
            chunk_idx=int(ex.get("chunk_idx") or 0),
            row_index=i,
            system_prompt=system_prompt,
        ))

    if not cases:
        return {"error": "no valid cases could be extracted", "skipped": skipped}

    # Sampling: explicit opt-in only (max_cases set AND smaller than the pool).
    total_cases = len(cases)
    coverage = "full"
    sampled = False
    if max_cases is not None and 0 < max_cases < total_cases:
        import random

        idx = sorted(random.Random(sample_seed).sample(range(total_cases), max_cases))
        cases = [cases[j] for j in idx]
        coverage = "sampled"
        sampled = True

    for c in cases:
        categories[c.category] = categories.get(c.category, 0) + 1
        d, _hint = _analyze_difficulty(c.question, c.correct_answer)
        difficulty[d] = difficulty.get(d, 0) + 1

    # Save suite
    os.makedirs(output_dir, exist_ok=True)
    if not suite_name:
        suite_name = Path(data_path).stem
    if sampled and "-sampled" not in suite_name:
        suite_name = f"{suite_name}-sampled{len(cases)}of{total_cases}"

    suite_path = os.path.join(output_dir, f"suite_{suite_name}.json")

    suite_data = []
    for c in cases:
        suite_data.append({
            "name": c.name,
            "category": c.category,
            "question": c.question,
            "correct_answer": c.correct_answer,
            "judge_hint": c.context,
            "source_id": c.source_id,
            "chunk_idx": c.chunk_idx,
            "row_index": c.row_index,
            "system_prompt": c.system_prompt,
        })

    with open(suite_path, "w") as f:
        json.dump({
            "meta": {
                "coverage": coverage,
                "dataset_count": len(examples),
                "pool_count": total_cases,
                "case_count": len(cases),
                "with_context_count": sum(1 for c in cases if c.system_prompt),
                "sample_size": len(cases) if sampled else None,
                "sample_seed": sample_seed if sampled else None,
            },
            "cases": suite_data,
        }, f, indent=2)

    return {
        "suite_path": suite_path,
        "suite_name": suite_name,
        "case_count": len(cases),
        "with_context_count": sum(1 for c in cases if c.system_prompt),
        "skipped": skipped,
        "invalid_lines": invalid_lines,
        "truncated": 0,
        "coverage": coverage,
        "dataset_count": len(examples),
        "pool_count": total_cases,
        "sample_seed": sample_seed if sampled else None,
        "source_ids": sorted({c.source_id for c in cases if c.source_id}),
        "categories": categories,
        "difficulty": difficulty,
    }


def _preference_messages(value: object, string_role: str) -> list[dict]:
    """Normalize one standard or conversational DPO field for quiz extraction."""
    if isinstance(value, str):
        return [{"role": string_role, "content": value}]
    if isinstance(value, list):
        return [message for message in value if isinstance(message, dict)]
    return []


def _categorize(question: str, answer: str) -> str:
    """Categorize a Q&A pair by its question type.

    Uses keyword signals from the question to determine what kind of
    reasoning is required to answer correctly.
    """
    q = question.lower()
    a = answer.lower()

    # Code/programming questions
    if any(k in q for k in ["code", "function", "script", "write a", "program", "implement", "python", "javascript"]):
        return "code"

    # Numeric/factual questions (who, what age, how many)
    if any(k in q for k in ["how many", "age", "population", "number of"]):
        return "numeric"

    # Factual recall (who is, what is the name of)
    if any(k in q for k in ["who is", "who are", "name of", "what is the name"]):
        return "factual"

    # Reasoning (why, how come, reason)
    if any(k in q for k in ["why", "reason", "how come", "cause of", "explain why"]):
        return "reasoning"

    # Process/procedure (how to, steps, way to)
    if any(k in q for k in ["how to", "steps to", "process of", "way to", "method for"]):
        return "process"

    # Temporal (when, what year, what age, during which)
    if any(k in q for k in ["when", "what year", "what age", "during which", "how long ago"]):
        return "temporal"

    # Spatial/location (where, location, region)
    if any(k in q for k in ["where", "location", "region", "place", "city of", "located"]):
        return "spatial"

    # Comparison (compare, difference, versus, vs)
    if any(k in q for k in ["compare", "difference", "versus", "vs ", "contrast"]):
        return "comparison"

    # Definition/explanation (what is, what are, define, explain)
    if any(k in q for k in ["what is", "what are", "what's", "define", "explain", "describe"]):
        return "knowledge"

    # Code in answer → code category
    if any(k in a for k in ["```", "def ", "class ", "function", "import ", "var ", "const "]):
        return "code"

    return "general"


def _analyze_difficulty(question: str, answer: str) -> tuple[str, str]:
    """Analyze answer characteristics to determine difficulty and judging strategy.

    Returns:
        (difficulty, judge_hint)

    difficulty: "easy" | "medium" | "hard"
    judge_hint: instruction for the judge on how to evaluate this case
    """
    answer_len = len(answer)
    # Short factual answers — exact match or close paraphrase
    if answer_len <= 30:
        return "easy", "exact_match: The model should give a specific, short factual answer. Accept exact match or very close paraphrase."

    # Medium — key facts must match, wording can differ
    if answer_len <= 100:
        return "medium", "key_facts: The model should include the key facts from the correct answer. Wording can differ but facts must be present."

    # Long explanation — semantic similarity matters
    return "hard", "semantic: The model should convey the same meaning and cover the main points. Exact wording is not required but core concepts must be present."


def _slugify(text: str) -> str:
    """Convert text to a safe filename slug."""
    text = text.lower().strip()
    text = re.sub(r'[^a-z0-9]+', '_', text)
    text = text.strip('_')
    return text[:60]
