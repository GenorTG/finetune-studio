"""Shared fakes for the preference-pair tests: a deterministic stand-in for the helper model + seeded pairs."""
from __future__ import annotations

import re
import time
from typing import Any

from finetune_studio import db
from finetune_studio.data.fs import qa as qa_fs

NAMES = ("quasar", "nebula", "pulsar", "vortex", "zenith", "meteor", "aurora", "cosmos", "photon", "ion")


def chunk_for(i: int) -> str:
    n = NAMES[i % len(NAMES)] + str(i)
    return f"The {n} router ships with four gigabit ethernet ports and a two year warranty."


def make_items(count: int = 8) -> list[dict[str, Any]]:
    """Deduplicated-pair-shaped dicts, one distinct subject per pair."""
    out = []
    for i in range(count):
        n = NAMES[i % len(NAMES)] + str(i)
        out.append({"id": f"p{i}", "source_id": f"s{i % 2}", "chunk_idx": i, "chunk_text": chunk_for(i),
                    "question": f"How many gigabit ethernet ports does the {n} router have?",
                    "answer": f"The {n} router has four gigabit ethernet ports.",
                    "category": "source-grounded", "status": "approved"})
    return out


class FakeModel:
    """Answers the three prompt shapes the builder sends; each behaviour is overridable per test."""

    def __init__(self, *, hallucination: str | None = None, verify: str = "NO",
                 fabricate: str = "The retail price is $499 and it ships worldwide.") -> None:
        self.hallucination, self.verify, self.fabricate = hallucination, verify, fabricate
        self.calls: list[str] = []

    def __call__(self, prompt: str) -> str:
        self.calls.append(prompt)
        if prompt.startswith("Below is a passage"):
            name = re.search(r"The (\w+) router", prompt).group(1)  # type: ignore[union-attr]
            return f'"What is the retail price of the {name} router?"'
        if prompt.startswith("Passage:"):
            return self.verify
        if prompt.startswith("Answer the question in one or two"):
            return self.fabricate
        # a bare question: what a model says without the source
        if self.hallucination is not None:
            return self.hallucination
        name = re.search(r"does the (\w+) router", prompt).group(1)  # type: ignore[union-attr]
        return f"The {name} has eight copper jacks, a metal shell and was designed in Sweden in 2019."


def seed_project(name: str = "pref-docs", count: int = 8, status: str = "approved") -> dict[str, Any]:
    proj = db.create_project(name, base_model="x/test")
    for item in make_items(count):
        qa_fs.write_qa_pair(proj["id"], {**item, "status": status, "created_at": time.time()})
    return proj
