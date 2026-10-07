"""Preference-pair builder (data/prep/preference.py): kinds, quality gates, determinism."""
from __future__ import annotations

import pytest

from finetune_studio.data.prep import preference as pref
from finetune_studio.training.data import format_for_preference
from tests.preference_fakes import FakeModel, chunk_for, make_items


def _build(items=None, model=None, **kw):
    return pref.build_preference_rows(items or make_items(), model or FakeModel(), **kw)


# ── both kinds, contract ───────────────────────────────────────────────

def test_builds_both_kinds_in_the_trl_contract() -> None:
    res = _build(max_pairs=8, seed=1)
    assert res.report.by_kind == {"hallucination": 4, "abstain": 4}
    format_for_preference(res.rows)  # the Training route's own validator accepts every row
    for row in res.rows:
        assert set(row) == {"prompt", "chosen", "rejected", "meta"}
        assert [m["role"] for m in row["prompt"]] == ["user"]
        assert row["chosen"][0]["role"] == row["rejected"][0]["role"] == "assistant"
        assert set(row["meta"]) == {"source_id", "chunk_idx", "kind", "origin"}


def test_hallucination_chosen_is_the_approved_answer_and_rejected_comes_from_the_model() -> None:
    items = make_items(4)
    res = _build(items, FakeModel(), kinds=("hallucination",), max_pairs=4)
    by_q = {i["question"]: i["answer"] for i in items}
    assert len(res.rows) == 4
    for row in res.rows:
        assert row["chosen"][0]["content"] == by_q[row["prompt"][0]["content"]]
        assert "eight copper jacks" in row["rejected"][0]["content"]


def test_abstain_pairs_ask_something_the_corpus_never_says() -> None:
    res = _build(kinds=("abstain",), max_pairs=4)
    assert len(res.rows) == 4
    corpus = " ".join(chunk_for(i) for i in range(8)).lower()
    for row in res.rows:
        assert row["chosen"][0]["content"] in pref.ABSTAIN_ANSWERS
        assert "retail price" in row["prompt"][0]["content"]
        assert "price" not in corpus
        assert "$499" in row["rejected"][0]["content"]


def test_abstain_question_sees_the_chunk_but_the_confident_answer_prompt_does_not() -> None:
    model = FakeModel()
    _build(model=model, kinds=("abstain",), max_pairs=2)
    confident = [c for c in model.calls if c.startswith("Answer the question")]
    assert confident and all("gigabit" not in c for c in confident)


# ── quality gates ──────────────────────────────────────────────────────

def test_rejected_equal_to_chosen_is_dropped() -> None:
    items = make_items(3)

    def echo(prompt: str) -> str:
        return next(i["answer"] for i in items if i["question"] == prompt)

    res = _build(items, echo, kinds=("hallucination",), max_pairs=3)
    assert res.rows == [] and res.report.dropped["hallucination"] == {"identical": 3}  # 3 candidates, all identical


def test_a_rejected_answer_as_grounded_as_the_chosen_one_teaches_nothing() -> None:
    items = make_items(3)
    # model "hallucinates" a paraphrase that is just as supported by the chunk
    res = _build(items, FakeModel(hallucination="It has four gigabit ethernet ports and a warranty."),
                 kinds=("hallucination",), max_pairs=3)
    assert res.rows == []
    assert set(res.report.dropped["hallucination"]) <= {"not_less_grounded", "near_duplicate"}


@pytest.mark.parametrize("refusal", ["I don't know that.", "As an AI I cannot say.", "I'm not sure about this one."])
def test_refusal_as_rejected_is_dropped_in_hallucination_pairs(refusal: str) -> None:
    res = _build(model=FakeModel(hallucination=refusal), kinds=("hallucination",), max_pairs=4)
    assert res.rows == [] and set(res.report.dropped["hallucination"]) == {"refusal_as_rejected"}


def test_empty_rejected_is_dropped() -> None:
    res = _build(model=FakeModel(hallucination="   "), kinds=("hallucination",), max_pairs=3)
    assert res.rows == [] and set(res.report.dropped["hallucination"]) == {"empty_rejected"}


def test_abstain_drops_hedged_fabrication_and_answerable_questions() -> None:
    hedged = _build(model=FakeModel(fabricate="I'm not sure, sorry."), kinds=("abstain",), max_pairs=3)
    assert hedged.rows == [] and set(hedged.report.dropped["abstain"]) == {"refusal_as_rejected"}
    # the model-check says the passage DOES answer it → over-refusal risk → dropped
    answerable = _build(model=FakeModel(verify="YES, it does."), kinds=("abstain",), max_pairs=3)
    assert answerable.rows == [] and set(answerable.report.dropped["abstain"]) == {"answerable_by_model_check"}


def test_abstain_question_gate_reasons() -> None:
    corpus = pref.content_tokens(chunk_for(0))
    chunk = chunk_for(0)
    gate = pref.abstain_question_gate
    assert gate("What is the retail price of the quasar0 router?", chunk, corpus, set()) is None
    assert gate("not a question", chunk, corpus, set()) == "malformed_question"
    assert gate("What is the retail price of the sandwich?", chunk, corpus, set()) == "off_subject"
    assert gate("What gigabit ethernet ports does the quasar0 router have?", chunk, corpus, set()) == "no_novel_detail"
    q = "What is the retail price of the quasar0 router?"
    assert gate(q, chunk, corpus, {pref.normalize_question(q)}) == "duplicate_question"


def test_prompts_are_deduplicated() -> None:
    items = make_items(3)
    items[1]["question"] = items[0]["question"]  # same question, different chunk (export dedupe keeps both)
    res = _build(items, FakeModel(hallucination="Six jacks, nope."), kinds=("hallucination",), max_pairs=3)
    keys = [pref.normalize_question(r["prompt"][0]["content"]) for r in res.rows]
    assert len(res.rows) == 2 and len(set(keys)) == 2 and res.report.deduped_prompts == 1


def test_model_failing_repeatedly_aborts_instead_of_shipping_a_thin_dataset() -> None:
    def boom(prompt: str) -> str:
        raise RuntimeError("helper unloaded")

    with pytest.raises(pref.GenerationFailed, match="helper unloaded"):
        _build(model=boom, kinds=("hallucination",), max_pairs=5)


def test_one_transient_model_error_is_skipped_and_counted() -> None:
    inner, state = FakeModel(), {"n": 0}

    def flaky(prompt: str) -> str:
        state["n"] += 1
        if state["n"] == 1:
            raise RuntimeError("blip")
        return inner(prompt)

    res = _build(model=flaky, kinds=("hallucination",), max_pairs=3)
    assert res.report.dropped["hallucination"]["generation_error"] == 1 and len(res.rows) == 3


# ── length bias, split, determinism, quotas ────────────────────────────

def test_length_stats_warn_when_chosen_is_systematically_longer() -> None:
    def rows(c: str, r: str) -> list[dict]:
        return [{"chosen": [{"content": c}], "rejected": [{"content": r}]}] * 4

    stats = pref.length_stats(rows("x" * 300, "y" * 100))
    assert stats.ratio == 3.0 and stats.mean_chosen_chars == 300 and stats.warning
    balanced = pref.length_stats(rows("x" * 100, "y" * 110))
    assert balanced.warning is None
    assert pref.length_stats([]).ratio == 0.0


def test_a_long_chosen_short_rejected_build_reports_the_warning() -> None:
    res = _build(model=FakeModel(hallucination="Six jacks, nope."), kinds=("hallucination",), max_pairs=4)
    assert res.report.length.ratio >= pref.LENGTH_RATIO_WARN and res.report.length.warning


def test_split_is_seeded_deterministic_and_disjoint() -> None:
    rows = _build(max_pairs=8).rows
    a = pref.split_preference_rows(rows, 0.25, seed=7)
    assert a == pref.split_preference_rows(rows, 0.25, seed=7)
    train, val = a
    assert len(train) == 6 and len(val) == 2
    assert not {r["prompt"][0]["content"] for r in train} & {r["prompt"][0]["content"] for r in val}
    assert pref.split_preference_rows(rows, 0.25, seed=8) != a


def test_same_seed_same_bytes_and_different_seed_different_sample() -> None:
    a, b = _build(max_pairs=4, seed=3), _build(max_pairs=4, seed=3)
    assert a.rows == b.rows
    big = make_items(10)
    s1 = _build(big, max_pairs=3, seed=1, kinds=("hallucination",)).rows
    s2 = _build(big, max_pairs=3, seed=2, kinds=("hallucination",)).rows
    assert s1 != s2


def test_max_pairs_caps_the_total_and_counts_attempts() -> None:
    res = _build(max_pairs=5)
    assert len(res.rows) == 5 and res.report.by_kind == {"hallucination": 3, "abstain": 2}
    assert res.report.attempted["hallucination"] >= 3


def test_pairs_without_a_stored_chunk_are_skipped_and_reported() -> None:
    items = make_items(4)
    items[0]["chunk_text"] = ""
    res = _build(items, max_pairs=4, kinds=("hallucination",))
    assert res.report.skipped_no_chunk == 1 and res.report.candidates == 3


def test_progress_callback_reports_attempts_and_kept() -> None:
    seen: list[dict] = []
    _build(max_pairs=4, progress=seen.append)
    assert seen and seen[-1]["attempted"] <= seen[-1]["planned"]
    assert sum(seen[-1]["kept"].values()) == 4


@pytest.mark.parametrize("kinds, n", [((), 5), (("style",), 5), (("abstain",), 0)])
def test_bad_arguments_are_rejected(kinds, n) -> None:
    with pytest.raises(ValueError):
        _build(kinds=kinds, max_pairs=n)
