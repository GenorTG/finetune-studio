"""Exhaustive mining: statements, coverage maths, gap rounds, extractive fallback, determinism, grounding."""
from __future__ import annotations

import json
from typing import Any

import pytest

from finetune_studio.data.prep import exhaustive as ex
from finetune_studio.data.prep.parsers import parse_qa_json
from finetune_studio.data.prep.qa_validate import validate_qa_pair

CHUNK = (
    "Annual leave. Full-time staff accrue 1.75 days of leave per month in their first three years.\n"
    "After year three the accrual rises to 2.1 days. Unused leave above 5 days expires on 31 March.\n"
    "Sick pay is paid at 80% for the first 14 days by Norda Med."
)

TABLE = (
    "=== Sheet: Reefer rates (3 rows) ===\n"
    "Lane code | Route | 20' EUR/day | Free days\n"
    "RC-01 | Gdynia - Hamburg | 38.5 | 3\n"
    "RC-02 | Gdynia - Rotterdam | 41.75 | 3"
)


def test_distinctive_tokens_pick_numbers_codes_and_names() -> None:
    toks = ex.distinctive_tokens("Contact Aurelia Hartwig on ext. 4120 about RC-01 at 38.50 EUR by 31 March.")
    assert {"aurelia", "hartwig", "4120", "rc-01", "38.5", "31"} <= toks
    assert "contact" not in toks and "about" not in toks


def test_numbers_compare_by_value() -> None:
    assert ex.canon("38.50") == ex.canon("38.5") == "38.5"
    assert ex.canon("41.00") == "41"


def test_statements_split_prose_and_attach_table_headers() -> None:
    prose = ex.split_statements(CHUNK)
    assert len(prose) == 4 and all(s.kind == "prose" for s in prose)
    rows = [s for s in ex.split_statements(TABLE) if s.kind == "table"]
    assert rows[0].text.startswith("Lane code") and rows[0].header == ""
    assert rows[1].header.startswith("Lane code") and rows[1].text.startswith("RC-01")


def test_a_chunk_that_starts_mid_table_uses_the_carried_header() -> None:
    head = "Lane code | Route | 20' EUR/day | Free days"
    rows = ex.split_statements("RC-03 | Gdynia - Gothenburg | 44.2 | 2", carried_header=head)
    assert rows[0].header == head
    assert ex.table_header_before([TABLE, "RC-03 | x | 1 | 2"], 1).startswith("Lane code")


def test_coverage_is_measured_by_distinctive_tokens_in_answers() -> None:
    statements = ex.split_statements(CHUNK)
    answers = ["Full-time staff accrue 1.75 days of leave per month in their first three years.",
               "Sick pay is 80% for 14 days via Norda Med."]
    open_ = ex.uncovered(statements, answers)
    assert {s.text.split()[0] for s in open_} == {"After", "Unused"}
    covered, total = ex.fact_coverage(statements, answers)
    assert (covered, total) == (2, 4)


def test_parse_salvages_a_reply_cut_off_by_the_token_limit() -> None:
    raw = '[{"q": "Q one?", "a": "A one 1."}, {"q": "Q two?", "a": "A two 2."}, {"q": "Q three?", "a": "A thr'
    assert [p["q"] for p in parse_qa_json(raw, 50)] == ["Q one?", "Q two?"]


def _driver(replies: list[str]):
    calls: list[list[dict[str, str]]] = []

    def chat(messages: list[dict[str, str]]) -> str:
        calls.append(messages)
        return replies[min(len(calls) - 1, len(replies) - 1)]

    return chat, calls


def _accept(pairs: list[dict[str, str]], chunk: str) -> list[dict[str, str]]:
    return [p for p in pairs if validate_qa_pair(p["q"], p["a"], chunk).accepted]


def _pair(q: str, a: str) -> dict[str, str]:
    return {"q": q, "a": a}


def test_gap_round_covers_what_the_first_pass_missed() -> None:
    first = json.dumps([_pair("How much leave do full-time staff accrue per month early on?",
                              "Full-time staff accrue 1.75 days of leave per month in their first three years."),
                        _pair("What is the sick pay rate and who pays it?", "Sick pay is paid at 80% for the first 14 days by Norda Med.")])
    second = json.dumps([_pair("What does the accrual rise to after year three?", "After year three the accrual rises to 2.1 days."),
                         _pair("When does unused leave above 5 days expire?", "Unused leave above 5 days expires on 31 March.")])
    chat, calls = _driver([first, second])
    out = ex.mine_chunk(CHUNK, chat=chat, parse=lambda r: parse_qa_json(r, 400), accept=_accept, title="Employee handbook",
                        section="Annual leave")
    assert len(calls) == 2 and out.rounds == 2
    assert out.covered == out.statements == 4 and out.extractive == 0
    assert [o for _, o in out.pairs] == ["model", "model", "model_gap", "model_gap"]
    assert "NOT covered" in calls[1][1]["content"] and "After year three" in calls[1][1]["content"]


def test_extractive_fallback_closes_what_the_model_never_writes_and_keeps_table_headers() -> None:
    chat, _ = _driver(["[]"])
    out = ex.mine_chunk(TABLE, chat=chat, parse=lambda r: parse_qa_json(r, 400), accept=_accept, title="Rate card",
                        section="Reefer rates")
    assert out.covered == out.statements and out.extractive >= 2
    rc02 = next(p for p, o in out.pairs if "RC-02" in p["q"])
    assert "41.75" in rc02["a"] and "Free days: 3" in rc02["a"]
    assert all(o == "extractive_gap" for _, o in out.pairs)


def test_a_failing_helper_call_does_not_lose_the_chunk() -> None:
    def boom(_m: list[dict[str, str]]) -> str:
        raise RuntimeError("helper offline")

    out = ex.mine_chunk(CHUNK, chat=boom, parse=lambda r: [], accept=_accept, title="Handbook", section="Leave")
    assert out.covered == out.statements and out.extractive >= 3


def test_invented_numbers_are_rejected_by_the_gate() -> None:
    ok = validate_qa_pair("How much leave accrues monthly?", "Staff accrue 1.75 days of leave per month.", CHUNK)
    bad = validate_qa_pair("How much leave accrues monthly?", "Staff accrue 2.5 days of leave per month.", CHUNK)
    assert ok.accepted and "ungrounded_value" in bad.reasons


@pytest.mark.parametrize("mode", ["exhaustive", "sampled", "nonsense"])
def test_runner_modes(mode: str) -> None:
    from finetune_studio.data.prep.runner import DataPrepRunner

    r: Any = DataPrepRunner("pid", b"x", "f.txt", mode=mode)
    assert r.mode == (mode if mode in ("exhaustive", "sampled") else "exhaustive")
