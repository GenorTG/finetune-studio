"""Dataset health check + duplicate removal (training page).

Gap (UI coverage audit 2026-10-03): dataset quality checks existed only behind
CLI / zero-caller routes, and the existing analyzer gave misleading advice
(PL/EN "balance" for any monolingual set, "I don't know" as a hallucination
risk, non-``messages`` trainable rows as format errors). This check reports
what actually breaks or weakens training, per dataset, in plain language.
"""

from __future__ import annotations

import json
from pathlib import Path

from finetune_studio.data.dataset_health import check_dataset, dedupe_dataset
from tests import test_training_start_guard as _guard

fake_home = _guard.fake_home


def _qa(q: str, a: str) -> dict:
    return {"messages": [{"role": "user", "content": q}, {"role": "assistant", "content": a}]}


def _write(tmp_path: Path, rows: list, name: str = "d.jsonl") -> Path:
    p = tmp_path / name
    p.write_text("".join((r if isinstance(r, str) else json.dumps(r)) + "\n" for r in rows), encoding="utf-8")
    return p


def _titles(report: dict) -> dict[str, dict]:
    return {i["code"]: i for i in report["issues"]}


def _clean(n: int) -> list[dict]:
    return [_qa(f"What is item {i}?", f"Item {i} is a well documented thing.") for i in range(n)]


def test_clean_dataset_has_no_issues(tmp_path: Path) -> None:
    rep = check_dataset(_write(tmp_path, _clean(40)))
    assert rep["verdict"] == "ok", rep["issues"]
    assert rep["examples"] == 40 and rep["trainable"] == 40
    assert rep["holdout"] == 4  # split_data keeps 90%: 40 - int(36.0)


def test_all_trainable_shapes_are_accepted(tmp_path: Path) -> None:
    rows = _clean(30) + [
        {"conversations": [{"from": "human", "value": "Hi there?"}, {"from": "gpt", "value": "Hello, friend."}]},
        {"prompt": "Name a colour?", "completion": "Blue is a colour."},
    ]
    rep = check_dataset(_write(tmp_path, rows))
    assert "untrainable" not in _titles(rep), rep["issues"]


def test_errors_name_their_lines(tmp_path: Path) -> None:
    rows = [*_clean(30), "{not json", {"foo": "bar"}, _qa("Empty answer?", "   ")]
    issues = _titles(check_dataset(_write(tmp_path, rows)))
    assert issues["invalid_json"]["severity"] == "error" and issues["invalid_json"]["lines"] == [31]
    assert issues["untrainable"]["lines"] == [32]
    assert issues["no_answer"]["lines"] == [33]


def test_duplicates_and_conflicting_answers(tmp_path: Path) -> None:
    rows = [*_clean(30), _qa("What is item 1?", "Item 1 is a well documented thing."),
            _qa("what is   ITEM 2?", "Something else entirely.")]
    issues = _titles(check_dataset(_write(tmp_path, rows)))
    assert issues["duplicates"]["count"] == 1 and issues["duplicates"]["lines"] == [31]
    assert issues["duplicates"]["action"] == "dedup"
    assert issues["conflicting_answers"]["count"] == 1
    assert issues["conflicting_answers"]["lines"] == [3, 32]


def test_small_dataset_warns_about_tiny_holdout(tmp_path: Path) -> None:
    issues = _titles(check_dataset(_write(tmp_path, _clean(8))))
    assert issues["small_dataset"]["severity"] == "warning"


def test_short_answers_flagged_and_dont_know_is_not(tmp_path: Path) -> None:
    rows = _clean(20) + [_qa(f"Short {i}?", "Yes.") for i in range(10)]
    rows += [_qa(f"Unknown {i}?", "I don't know; that is not in the documents.") for i in range(10)]
    issues = _titles(check_dataset(_write(tmp_path, rows)))
    assert issues["short_answers"]["count"] == 10
    assert not any("know" in i["title"].lower() for i in issues.values())


def test_monolingual_dataset_gets_no_language_advice(tmp_path: Path) -> None:
    rows = [_qa(f"Czym jest żółw {i}?", f"Żółw {i} to gad, który żyje długo.") for i in range(30)]
    issues = check_dataset(_write(tmp_path, rows))["issues"]
    assert not any("polish" in i["title"].lower() or "english" in i["title"].lower() for i in issues)


def test_dedupe_keeps_first_occurrence_and_original_lines(tmp_path: Path) -> None:
    src = _write(tmp_path, [*_clean(3), _qa("What is item 0?", "Item 0 is a well documented thing."), "{bad"])
    dst = tmp_path / "out.jsonl"
    kept, removed = dedupe_dataset(src, dst)
    assert (kept, removed) == (4, 1)
    lines = dst.read_text().splitlines()
    assert lines[-1] == "{bad"  # unreadable lines are not the dedup's business; kept verbatim
    assert len(lines) == 4


# ── routes ───────────────────────────────────────────────────────────────────


def _dataset(client, rows: list) -> tuple[str, dict]:
    pid = _guard._project(client)
    body = "".join(json.dumps(r) + "\n" for r in rows).encode()
    r = client.post(f"/api/projects/{pid}/datasets/upload", files={"file": ("qa.jsonl", body, "application/jsonl")})
    assert r.status_code == 200, r.text
    return pid, r.json()


def test_health_route(client, fake_home):
    pid, ds = _dataset(client, [*_clean(30), _qa("What is item 1?", "Item 1 is a well documented thing.")])
    r = client.get(f"/api/projects/{pid}/datasets/{ds['id']}/health")
    assert r.status_code == 200, r.text
    assert _titles(r.json())["duplicates"]["count"] == 1


def test_health_route_scoped_to_project(client, fake_home):
    _pid, ds = _dataset(client, _clean(3))
    other = _guard._project(client)
    assert client.get(f"/api/projects/{other}/datasets/{ds['id']}/health").status_code == 404


def test_dedup_route_registers_new_dataset_and_keeps_original(client, fake_home):
    pid, ds = _dataset(client, [*_clean(30), _qa("What is item 1?", "Item 1 is a well documented thing.")])
    r = client.post(f"/api/projects/{pid}/datasets/{ds['id']}/dedup")
    assert r.status_code == 200, r.text
    new = r.json()
    assert new["id"] != ds["id"] and new["qa_count"] == 30 and new["removed"] == 1
    assert Path(ds["data_path"]).read_text().count("\n") == 31  # original untouched
    names = {d["name"] for d in client.get(f"/api/projects/{pid}/datasets").json()["datasets"]}
    assert {ds["name"], new["name"]} <= names


def test_dedup_route_with_nothing_to_remove_is_400(client, fake_home):
    pid, ds = _dataset(client, _clean(5))
    r = client.post(f"/api/projects/{pid}/datasets/{ds['id']}/dedup")
    assert r.status_code == 400 and "no duplicates" in r.json()["error"]


def test_training_page_renders_health_panel() -> None:
    from finetune_studio import webui

    html = (Path(webui.__file__).parent / "templates" / "project_training.html").read_text()
    assert 'id="dataset-health"' in html
    assert "/datasets/${encodeURIComponent(did)}/${what}" in html
    assert 'healthApi(did, "health")' in html
    assert 'healthApi(btn.dataset.healthDedup, "dedup")' in html
    assert "window.notify" not in html  # the toast helper is window.fts.notify
