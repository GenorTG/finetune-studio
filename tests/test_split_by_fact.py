"""Train/validation split keeps a fact's rows together (training.data.split_data, sft_validation_rows).

A study set writes each fact 4-5 ways under one ``pair_id``; a row-by-row split put rewordings of
held-out facts into training, so "held-out" scores measured memorisation.
"""
from __future__ import annotations

import random

from finetune_studio.training.data import (
    format_for_sft,
    sft_validation_rows,
    split_data,
)


def _row(q: str, a: str, pair: str | None = None) -> dict:
    row = {"conversations": [{"from": "human", "value": q}, {"from": "gpt", "value": a}]}
    if pair:
        row["pair_id"] = pair
    return row


def _legacy_split(data: list, train_ratio: float = 0.9, seed: int = 42):
    shuffled = data.copy()
    random.Random(seed).shuffle(shuffled)
    split = min(max(1, int(len(shuffled) * train_ratio)), len(shuffled) - 1)
    return shuffled[:split], shuffled[split:]


def test_rows_without_groups_split_exactly_as_before() -> None:
    rows = format_for_sft([_row(f"q{i}", f"a{i}") for i in range(50)])
    assert split_data(list(rows)) == _legacy_split(list(rows))


def test_rows_of_one_fact_never_straddle_the_split() -> None:
    raw = [_row(f"q{p}-{k}", f"a{p}", pair=f"p{p}") for p in range(40) for k in range(4)]
    train, val = split_data(format_for_sft(raw))
    assert len(val) == 4 * 4 and len(train) == 36 * 4  # 10 % of the FACTS, all their rows
    fact = lambda r: r["messages"][1]["content"]
    assert not {fact(r) for r in train} & {fact(r) for r in val}
    assert all("_split_group" not in r for r in train + val)


def test_validation_rows_are_the_trainers_own_slice() -> None:
    raw = [_row(f"q{p}-{k}", f"a{p}", pair=f"p{p}") for p in range(30) for k in range(3)]
    raw.insert(5, {"junk": True})  # skipped by the formatter, as in training
    val_raw = sft_validation_rows(raw)
    _train, val = split_data(format_for_sft(raw))
    assert [r["conversations"][0]["value"] for r in val_raw] == [r["messages"][0]["content"] for r in val]


def test_heldout_eval_uses_the_grouped_slice(tmp_path, monkeypatch) -> None:
    import json

    from finetune_studio.testing import training_eval

    path = tmp_path / "study.jsonl"
    raw = [_row(f"Question {p} form {k}?", f"Answer {p}.", pair=f"p{p}") for p in range(20) for k in range(4)]
    path.write_text("".join(json.dumps(r) + "\n" for r in raw), encoding="utf-8")
    monkeypatch.setattr(training_eval, "resolve_project_dataset",
                        lambda pid, did: {"id": "d", "name": "study", "data_path": str(path), "source": "x"})
    cases, meta = training_eval.build_heldout_eval("pid")
    assert meta.case_count == 8  # 2 of 20 facts x 4 forms
    trained = {r["messages"][1]["content"] for r in split_data(format_for_sft(raw))[0]}
    assert not {c.correct_answer for c in cases} & trained
