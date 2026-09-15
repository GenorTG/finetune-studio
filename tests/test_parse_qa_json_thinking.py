"""Regression: prep Q&A parsing must survive Qwen3's bare </think> preamble."""
from __future__ import annotations

from finetune_studio.data.prep.parsers import parse_qa_json


def test_bare_close_think_preamble_is_dropped() -> None:
    raw = (
        "The chunk mentions Eldrathane [the capital] so I should ask about it.\n"
        "</think>\n\n"
        '[{"q": "What is the capital of Velmaris?", "a": "Eldrathane"}]'
    )
    pairs = parse_qa_json(raw, 1)
    assert len(pairs) == 1
    assert "Eldrathane" in str(pairs[0])


def test_paired_think_block_still_stripped() -> None:
    raw = '<think>[draft]</think>[{"q": "Who founded the Ember College?", "a": "Pyra Vell"}]'
    pairs = parse_qa_json(raw, 1)
    assert len(pairs) == 1
    assert "Pyra Vell" in str(pairs[0])
