"""Replay (training/replay.py): general prompts answered by the base model, mixed into SFT, cached per base."""
from __future__ import annotations

import pytest
import torch

from finetune_studio.training import replay


def test_row_count_makes_the_requested_share_of_the_mix() -> None:
    assert replay.replay_row_count(7000, 0.3) == 3000  # 3000 / 10000 = 30 %
    assert replay.replay_row_count(7000, 0.0) == 0
    assert replay.replay_row_count(100000, 0.5) == replay.MAX_REPLAY_ROWS
    with pytest.raises(ValueError):
        replay.replay_row_count(10, 1.0)


def _dolly(n: int) -> list[dict]:
    rows = []
    for i in range(n):
        cat = "closed_qa" if i % 3 == 0 else "open_qa"
        rows.append({"instruction": f"Question {i}?", "context": "" if i % 6 == 0 else f"Context {i}.", "category": cat})
    return rows


def test_prompts_are_seeded_and_context_tasks_carry_their_context() -> None:
    data = _dolly(60)

    def loader(name, *, split, cache_dir):
        assert (name, split) == (replay.PROMPT_DATASET, "train")
        return data

    a = replay.load_prompts(10, seed=1, loader=loader)
    assert a == replay.load_prompts(10, seed=1, loader=loader) and len(a) == 10
    assert a != replay.load_prompts(10, seed=2, loader=loader)
    for p in a:
        i = int(p.split()[1].rstrip("?"))
        if i % 3 == 0:  # closed_qa: needs its context, rows without one are skipped
            assert i % 6 != 0 and f"Context {i}." in p


class _Tok:
    """Characters as tokens; id 1 ends a turn, 0 pads."""

    padding_side = "right"
    pad_token_id = 0
    eos_token_id = 1
    unk_token_id = None

    def get_vocab(self) -> dict:
        return {}

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False, **_):
        return "|".join(m["content"] for m in messages) + ">"

    def __call__(self, texts, return_tensors=None, padding=False, add_special_tokens=False):
        width = max(len(t) for t in texts)
        ids = torch.tensor([[0] * (width - len(t)) + [ord(c) for c in t] for t in texts])

        class Enc(dict):
            def to(self, _device):
                return self

        return Enc(input_ids=ids, attention_mask=(ids != 0).long())

    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(i) for i in ids)


class _Model:
    """Answers "ok"; a prompt containing LONG never ends its turn (hits the token limit)."""

    device = "cpu"
    training = True
    generation_config = None

    def eval(self) -> None:
        self.training = False

    def train(self) -> None:
        self.training = True

    def generate(self, input_ids, max_new_tokens, **_):
        rows = []
        for row in input_ids:
            text = "".join(chr(int(i)) for i in row if int(i))
            tail = [ord("x")] * 4 if "LONG" in text else [ord("o"), ord("k"), 1, 0]
            rows.append(row.tolist() + tail)
        return torch.tensor(rows)


def test_answers_drop_truncated_ones_and_restore_the_model() -> None:
    model, tok = _Model(), _Tok()
    seen = []
    out = replay.generate_answers(model, tok, ["hi", "LONG one", "bye"], batch_size=2,
                                  progress=lambda d, t: seen.append((d, t)))
    assert out == [("hi", "ok"), ("bye", "ok")]
    assert seen == [(2, 3), (3, 3)]
    assert model.training is True and tok.padding_side == "right"


def test_rows_are_cached_per_base_and_reused(tmp_path, monkeypatch) -> None:
    from finetune_studio.data.fs import paths

    monkeypatch.setattr(paths, "_ROOT", tmp_path)
    calls = []

    def fake_answers(model, tokenizer, prompts, **kw):
        calls.append(list(prompts))
        return [(p, f"answer to {p}") for p in prompts]

    monkeypatch.setattr(replay, "generate_answers", fake_answers)
    loader = lambda name, *, split, cache_dir: _dolly(200)
    rows = replay.replay_rows(None, None, model_path="/m/base", n=20, system_prompt="Be brief.",
                              prompt_loader=loader)
    assert len(rows) == 20 and rows[0]["messages"][0] == {"role": "system", "content": "Be brief."}
    assert rows[0]["messages"][-1]["content"].startswith("answer to ")
    again = replay.replay_rows(None, None, model_path="/m/base", n=20, system_prompt="Be brief.",
                               prompt_loader=loader)
    assert again == rows and len(calls) == 1  # second run: cache only
    replay.replay_rows(None, None, model_path="/m/other", n=5, prompt_loader=loader)
    assert len(calls) == 2  # another base model answers for itself


def test_engine_mixes_replay_only_for_sft(monkeypatch) -> None:
    from finetune_studio.training.engine import TrainingConfig, TrainingEngine

    got = {}

    def fake_rows(model, tokenizer, *, model_path, n, system_prompt, progress):
        got.update(n=n, system_prompt=system_prompt)
        return [{"messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]}] * n

    monkeypatch.setattr(replay, "replay_rows", fake_rows)
    eng = TrainingEngine()
    eng.config = TrainingConfig(model_path="/m", replay_fraction=0.3)
    eng._bake_prompt = "SYS"
    chat = type("Tok", (), {"chat_template": "{{ messages }}"})()
    with pytest.raises(ValueError, match="chat template"):
        eng._replay(None, type("Bare", (), {"chat_template": None})(), 700)
    assert len(eng._replay(None, chat, 700)) == 300 and got == {"n": 300, "system_prompt": "SYS"}
    assert any("Replay: 300 general rows" in line for line in eng.state.log_lines)
    eng.config = TrainingConfig(model_path="/m", replay_fraction=0.3, training_mode="dpo")
    assert eng._replay(None, chat, 700) == []
    eng.config = TrainingConfig(model_path="/m")
    assert eng._replay(None, chat, 700) == []


@pytest.mark.parametrize(("value", "mode", "status"), [("30", "sft", None), ("60", "sft", 400),
                                                        ("x", "sft", 400), ("20", "kto", 400)])
def test_start_route_validates_replay_percent(client, tmp_path, monkeypatch, value, mode, status) -> None:
    from pathlib import Path

    import finetune_studio.webui.routes.training as tr
    from tests import test_training_start_guard as g

    eng = g._FakeEngine()
    monkeypatch.setattr(tr, "training_engine", eng)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    model = tmp_path / "m"
    model.mkdir()
    r = g._start(client, g._project(client), g._jsonl(tmp_path), str(model), training_mode=mode,
                 replay_percent=value)
    if status:
        assert r.status_code == status and "replay" in r.json()["detail"], r.text
    else:
        assert r.json().get("status") == "started", r.text
        assert eng.started["config"].replay_fraction == pytest.approx(0.3)


def test_dropped_answers_are_topped_up_in_rounds(tmp_path, monkeypatch) -> None:
    from finetune_studio.data.fs import paths

    monkeypatch.setattr(paths, "_ROOT", tmp_path)
    asked: list[str] = []

    def half_kept(model, tokenizer, prompts, **kw):
        asked.extend(prompts)
        return [(p, "a") for i, p in enumerate(prompts) if i % 2 == 0]

    monkeypatch.setattr(replay, "generate_answers", half_kept)
    rows = replay.replay_rows(None, None, model_path="/m/b", n=30, prompt_loader=lambda *a, **k: _dolly(400))
    assert len(rows) == 30
    assert len(asked) == len(set(asked))  # no prompt is asked twice
