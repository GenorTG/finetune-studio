"""Auto-generate benchmark suites from training data.

WHAT THIS FILE DOES
==================
After training, converts the training data (JSONL with Q&A pairs) into
a benchmark suite that tests exactly what the model was trained on.

WHY
===
If you train on Aethermere lore, the benchmark should test Aethermere lore —
not general AI knowledge. This module bridges that gap.

FLOW
======
1. Training completes → training data is a JSONL file with conversations
2. Each Q&A pair becomes a BenchmarkCase (question + correct answer)
3. Suite is saved as JSON alongside the training output
4. DB records the suite path so the UI can find it
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from finetune_studio.testing.suite import BenchmarkCase


def generate_suite_from_training_data(
    data_path: str,
    output_dir: str,
    suite_name: str | None = None,
    min_answer_length: int = 10,
    max_cases: int = 500,
) -> dict:
    """Convert a training JSONL file into a benchmark suite.

    Args:
        data_path: Path to JSONL training data (conversations format)
        output_dir: Where to save the suite JSON
        suite_name: Name for the suite (defaults to filename)
        min_answer_length: Skip answers shorter than this (chars)
        max_cases: Maximum number of cases to include

    Returns:
        {suite_path, case_count, skipped, categories}
    """
    data_path = str(data_path)
    output_dir = str(output_dir)

    if not os.path.isfile(data_path):
        return {"error": f"training data not found: {data_path}"}

    # Load training data
    examples = []
    with open(data_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                examples.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    if not examples:
        return {"error": "no valid examples in training data"}

    # Convert to benchmark cases
    cases = []
    skipped = 0
    categories: dict[str, int] = {}

    for i, ex in enumerate(examples):
        if len(cases) >= max_cases:
            break

        # Extract Q&A from conversations format
        conversations = ex.get("conversations", [])
        if len(conversations) < 2:
            skipped += 1
            continue

        question = conversations[0].get("value", "").strip()
        answer = conversations[1].get("value", "").strip()

        if not question or not answer:
            skipped += 1
            continue

        if len(answer) < min_answer_length:
            skipped += 1
            continue

        # Derive category from question content
        q_lower = question.lower()
        if any(k in q_lower for k in ["what is", "what are", "what's", "define", "explain"]):
            category = "knowledge"
        elif any(k in q_lower for k in ["why", "reason", "cause"]):
            category = "reasoning"
        elif any(k in q_lower for k in ["how", "steps", "process", "way to"]):
            category = "process"
        elif any(k in q_lower for k in ["who", "whom", "whose", "name of"]):
            category = "factual"
        elif any(k in q_lower for k in ["when", "time", "year", "date", "age"]):
            category = "temporal"
        elif any(k in q_lower for k in ["where", "location", "place", "region"]):
            category = "spatial"
        elif any(k in q_lower for k in ["compare", "difference", "versus", "vs"]):
            category = "comparison"
        else:
            category = "general"

        # Create a stable name from the question
        name = _slugify(question[:60])

        cases.append(BenchmarkCase(
            name=name,
            question=question,
            correct_answer=answer,
            category=category,
        ))

        categories[category] = categories.get(category, 0) + 1

    if not cases:
        return {"error": "no valid cases could be extracted", "skipped": skipped}

    # Save suite
    os.makedirs(output_dir, exist_ok=True)
    if not suite_name:
        suite_name = Path(data_path).stem

    suite_path = os.path.join(output_dir, f"suite_{suite_name}.json")

    suite_data = []
    for c in cases:
        suite_data.append({
            "name": c.name,
            "category": c.category,
            "question": c.question,
            "correct_answer": c.correct_answer,
        })

    with open(suite_path, "w") as f:
        json.dump(suite_data, f, indent=2)

    return {
        "suite_path": suite_path,
        "suite_name": suite_name,
        "case_count": len(cases),
        "skipped": skipped,
        "categories": categories,
    }


def _slugify(text: str) -> str:
    """Convert text to a safe filename slug."""
    import re
    text = text.lower().strip()
    text = re.sub(r'[^a-z0-9]+', '_', text)
    text = text.strip('_')
    return text[:60]
