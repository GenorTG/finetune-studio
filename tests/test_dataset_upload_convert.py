"""Dataset upload converts JSON/CSV into trainable JSONL instead of renaming it.

Regression (UI coverage audit 2026-10-03): ``POST /projects/{pid}/datasets/upload``
wrote the raw bytes of a ``.json`` / ``.csv`` file under a ``.jsonl`` name, so
training read garbage; and ``converter.csv_to_jsonl`` emitted user-only rows
(no answers), useless for SFT.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from finetune_studio.data.converter import (
    csv_to_jsonl,
    records_from_upload,
)
from finetune_studio.training.data import format_for_sft
from tests import test_training_start_guard as _guard

fake_home = _guard.fake_home

QA = [{"role": "user", "content": "What is X?"}, {"role": "assistant", "content": "X is Y."}]

# ── pure parsing ─────────────────────────────────────────────────────────────


def test_jsonl_passes_through() -> None:
    raw = (json.dumps({"messages": QA}) + "\n\n" + json.dumps({"prompt": "a", "completion": "b"}) + "\n").encode()
    recs = records_from_upload(raw, "d.jsonl")
    assert recs == [{"messages": QA}, {"prompt": "a", "completion": "b"}]


def test_json_array_and_wrapped_object() -> None:
    assert records_from_upload(json.dumps([{"messages": QA}]).encode(), "d.json") == [{"messages": QA}]
    wrapped = json.dumps({"data": [{"messages": QA}, {"messages": QA}]}).encode()
    assert len(records_from_upload(wrapped, "d.json")) == 2


def test_question_answer_rows_become_messages() -> None:
    raw = json.dumps([{"Question": "What is X?", "Answer": "X is Y."}]).encode()
    assert records_from_upload(raw, "d.json") == [{"messages": QA}]


def test_csv_question_answer_and_system() -> None:
    raw = b"question,answer,system\nWhat is X?,X is Y.,Be brief.\n"
    recs = records_from_upload(raw, "d.csv")
    assert recs == [{"messages": [{"role": "system", "content": "Be brief."}, *QA]}]


def test_csv_alpaca_instruction_input_output() -> None:
    raw = b"instruction,input,output\nTranslate,hello,hola\nSay hi,,hi\n"
    recs = records_from_upload(raw, "d.csv")
    assert recs[0]["messages"][0] == {"role": "user", "content": "Translate\n\nhello"}
    assert recs[0]["messages"][1] == {"role": "assistant", "content": "hola"}
    assert recs[1]["messages"][0]["content"] == "Say hi"


def test_csv_text_column_kept_as_text() -> None:
    assert records_from_upload(b"text\nhello world\n", "d.csv") == [{"text": "hello world"}]


def test_every_converted_record_is_trainable() -> None:
    raw = b"prompt,response\nWhat is X?,X is Y.\n"
    recs = records_from_upload(raw, "d.csv")
    assert format_for_sft(recs) == [{"messages": QA}]


@pytest.mark.parametrize(("raw", "name", "needle"), [
    (b"col_a,col_b\n1,2\n", "d.csv", "col_a"),          # unmappable columns are named
    (b'{"messages": []}\n{bad\n', "d.jsonl", "line 2"),  # bad JSONL line is located
    (b"not json", "d.json", "JSON"),
    (b"hello", "notes.txt", ".jsonl, .json or .csv"),
    (b"[]", "d.json", "no training examples"),
    (b"[1, 2]", "d.json", "object"),
])
def test_bad_uploads_raise_readable_errors(raw: bytes, name: str, needle: str) -> None:
    with pytest.raises(ValueError) as exc:
        records_from_upload(raw, name)
    assert needle in str(exc.value)


def test_cli_csv_to_jsonl_now_writes_answers(tmp_path: Path) -> None:
    src = tmp_path / "d.csv"
    src.write_text("question,answer\nWhat is X?,X is Y.\n", encoding="utf-8")
    out = tmp_path / "d.jsonl"
    csv_to_jsonl(str(src), str(out))
    assert json.loads(out.read_text().strip()) == {"messages": QA}


# ── route ────────────────────────────────────────────────────────────────────


def _project(client) -> str:
    return _guard._project(client)


def test_upload_csv_is_converted(client, fake_home):
    pid = _project(client)
    r = client.post(f"/api/projects/{pid}/datasets/upload",
                    files={"file": ("faq.csv", b"question,answer\nWhat is X?,X is Y.\n", "text/csv")})
    assert r.status_code == 200, r.text
    ds = r.json()
    path = Path(ds["data_path"])
    assert path.suffix == ".jsonl" and ds["qa_count"] == 1
    assert json.loads(path.read_text().strip()) == {"messages": QA}


def test_upload_unmappable_csv_is_400_and_writes_nothing(client, fake_home):
    pid = _project(client)
    r = client.post(f"/api/projects/{pid}/datasets/upload",
                    files={"file": ("x.csv", b"a,b\n1,2\n", "text/csv")})
    assert r.status_code == 400, r.text
    assert "a, b" in r.json()["error"]
    assert client.get(f"/api/projects/{pid}/datasets").json() == {"datasets": []}
