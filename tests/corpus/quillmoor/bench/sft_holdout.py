"""Create matched seen/unseen closed-book questions from the app's SFT fact split."""
from __future__ import annotations

import hashlib
import random
from typing import Any

TRAIN_SAMPLE_SEED = 20261029


def sft_fact_questions(training_rows: list[dict[str, Any]], canonical_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Pair a deterministic sample of seen facts with every held-out fact; reject fact leakage."""
    from finetune_studio.training.data import sft_validation_rows

    validation = sft_validation_rows(training_rows)
    validation_ids = {id(row) for row in validation}
    training = [row for row in training_rows if id(row) not in validation_ids]
    validation_facts = {str(row["fact"]) for row in validation if row.get("fact")}
    training_facts = {str(row["fact"]) for row in training if row.get("fact")}
    overlap = validation_facts & training_facts
    if overlap:
        raise ValueError(
            f"SFT validation split leaks {len(overlap)}/{len(validation_facts)} facts into training; "
            "use a fact-grouped dataset before scoring generalization"
        )
    if not validation_facts:
        raise ValueError("SFT validation split contains no fact IDs")
    if not training_facts:
        raise ValueError("SFT training split contains no fact IDs")

    canonical_by_fact: dict[str, dict[str, Any]] = {}
    for row in canonical_rows:
        fact = row.get("fact")
        if not isinstance(fact, str) or not fact:
            raise ValueError("canonical SFT rows must contain a non-empty fact ID")
        if fact in canonical_by_fact:
            raise ValueError(f"canonical SFT rows contain duplicate fact ID: {fact}")
        canonical_by_fact[fact] = row
    missing = (validation_facts | training_facts) - canonical_by_fact.keys()
    if missing:
        raise ValueError(f"{len(missing)} split facts are absent from the canonical training rows")

    sample_count = min(len(validation_facts), len(training_facts))
    sampled_training_facts = sorted(random.Random(TRAIN_SAMPLE_SEED).sample(sorted(training_facts), sample_count))
    questions: list[dict[str, Any]] = []
    for scope, fact_ids in (("sft-trained", sampled_training_facts), ("sft-heldout", sorted(validation_facts))):
        for fact in fact_ids:
            messages = canonical_by_fact[fact].get("messages") or []
            question = next((m.get("content") for m in messages if m.get("role") == "user"), None)
            answer = next((m.get("content") for m in messages if m.get("role") == "assistant"), None)
            if not isinstance(question, str) or not question.strip() or not isinstance(answer, str) or not answer.strip():
                raise ValueError(f"canonical SFT row has no usable question/answer for fact: {fact}")
            question_id = hashlib.sha256(fact.encode("utf-8")).hexdigest()[:20]
            questions.append({
                "id": f"{scope}-{question_id}",
                "type": scope,
                "question": question,
                "answer": answer,
                "accept": [],
                "wrong": [],
                "formats": [],
                "etype": "sft",
                "attr": "fact",
                "fact": fact,
            })
    return questions
