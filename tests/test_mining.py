"""Per-file mining (data.mining): inventory → mine → check → fix, stored rounds, coverage states, review loop, export gate.

The helper is a scripted fake (no model): it lists data points, writes pairs from the checklist it is given, and only
answers the table row in a FIX round, so every stage is exercised for real through the HTTP API and the scheduler threads.
"""
from __future__ import annotations

import json
import threading
import time
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from finetune_studio import db
from finetune_studio.config import settings
from finetune_studio.data.fs import qa as qa_fs
from finetune_studio.data.mining import inventory as inv
from finetune_studio.data.mining import statements as st
from finetune_studio.data.mining.coverage import file_status, item_states
from finetune_studio.webui.app import app

DOC = """Leave policy extract

Annual leave is 21 working days for employees with 0-2 years of service at Korvane.
During notice the Company may grant garden leave, in which case base pay continues in full.

Lane | Rate EUR
RC-01 | 38.50
RC-02 | 41.75
"""


class FakeHelper:
    """Answers by the kind of prompt it receives; records concurrency and every user prompt."""

    def __init__(self, delay: float = 0.0) -> None:
        self.prompts: list[str] = []
        self.delay = delay
        self.active = 0
        self.max_active = 0
        self.lock = threading.Lock()
        self.fail = False

    def __call__(self, messages: list[dict[str, str]], **_: Any) -> str:
        with self.lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            time.sleep(self.delay)
            if self.fail:
                raise RuntimeError("helper is down")
            system, user = messages[0]["content"], messages[-1]["content"]
            with self.lock:
                self.prompts.append(user)
            if "List ONLY the facts that need more than one part" in user:
                return json.dumps([{"fact": "Korvane grants garden leave with full base pay", "values": ["garden leave", "base pay continues in full"]}])
            if system.startswith("You list the data points"):
                return json.dumps([
                    {"fact": "Garden leave keeps base pay", "values": ["garden leave", "base pay continues in full"]},
                    {"fact": "An invented fact", "values": ["99 days of sabbatical"]},          # not in the text: dropped
                    {"fact": "RC-01 rate", "values": ["RC-01", "38.50"]},                       # only repeats a rule row: dropped
                ])
            pairs = []
            if "These data points of the passage are NOT stated" in user:
                for line in user.splitlines():                     # a page reference is not a fact: propose to skip it
                    num, _, rest = line.partition(". ")
                    if num.isdigit() and "PR-77" in rest:
                        pairs.append({"skip": int(num), "reason": "page footer"})
                if "RC-02" in user:
                    pairs.append({"q": "What is the rate of lane RC-02 in the Korvane leave policy extract?", "a": "Lane RC-02 costs EUR 41.75."})
                return json.dumps(pairs)
            pairs += [
                {"q": "How much annual leave do Korvane employees with 0-2 years of service get?",
                 "a": "Annual leave is 21 working days for employees with 0-2 years of service at Korvane."},
                {"q": "What happens to base pay during garden leave at Korvane?",
                 "a": "During notice the Company may grant garden leave, and base pay then continues in full."},
                {"q": "What is the rate of lane RC-01 in the Korvane leave policy extract?", "a": "Lane RC-01 costs EUR 38.50."},
            ]
            return json.dumps(pairs)
        finally:
            with self.lock:
                self.active -= 1


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[tuple[TestClient, FakeHelper]]:
    monkeypatch.setattr(settings, "db_path", str(tmp_path / "mining.db"))
    root = tmp_path / "fts"
    projects = root / "projects"
    projects.mkdir(parents=True)
    monkeypatch.setattr("finetune_studio.data.fs.paths._ROOT", root)
    monkeypatch.setattr("finetune_studio.data.fs.paths._PROJECTS", projects)
    monkeypatch.setattr(qa_fs, "project_dir", lambda pid: projects / pid)
    db.init_db()
    helper = FakeHelper()
    monkeypatch.setattr("finetune_studio.data.prep.generator.resolve_generator", lambda: helper)
    # The mining worker (a job; on a thread in tests) resolves the helper recorded on the job, not the web's.
    monkeypatch.setattr("finetune_studio.data.mining.seat.chat_for", lambda target: helper)
    monkeypatch.setattr("finetune_studio.models.helper.get_configured_helper_provider", lambda: {
        "id": "api-helper", "kind": "openai_compat", "model_id": "fake-model", "label": "Helper · API · fake",
        "extra": {"concurrency": 3, "body": {"reasoning_effort": "none"}}})
    yield TestClient(app), helper
    from finetune_studio.jobs import launch
    for j in launch.list_jobs("mining", running_only=True):
        launch.stop(j["id"])
        launch.wait(j["id"], timeout=30)


def _project_with_file(client: TestClient, text: str = DOC, name: str = "leave.txt") -> tuple[str, str]:
    pid = client.post("/api/projects", json={"name": f"mine-{uuid.uuid4().hex[:6]}", "base_model": "x"}).json()["id"]
    return pid, _upload(client, pid, name, text)


def _upload(client: TestClient, pid: str, name: str, text: str) -> str:
    """Upload through the file library; its parse runs as the response's background task. Returns the source id."""
    from finetune_studio.knowledge.sources import get_source

    r = client.post(f"/api/projects/{pid}/files/upload", files={"files": (name, text.encode())})
    assert r.status_code == 200, r.text
    sid = r.json()["report"][0]["source_id"]
    source = get_source(pid, sid)
    assert source is not None and source.ready, source
    return sid


def _wait(pid: str, timeout: float = 20.0) -> list[dict]:
    end = time.time() + timeout
    while time.time() < end:
        if not db.mining.list_jobs(pid, active=True):
            return db.mining.list_jobs(pid)
        time.sleep(0.05)
    raise AssertionError("mining jobs did not finish")


def _mine(client: TestClient, pid: str, sid: str, kind: str = "mine") -> dict:
    r = client.post(f"/api/projects/{pid}/mining/jobs", json={"source_ids": [sid], "kind": kind})
    assert r.status_code == 200, r.text
    jid = r.json()["jobs"][0]["id"]
    _wait(pid)
    return db.mining.get_job(jid)  # type: ignore[return-value]


# ── the whole loop through the API ──────────────────────────────────────────────────────────────────────────────────

def test_upload_only_parses_and_mining_is_an_explicit_job(env) -> None:
    client, helper = env
    pid, sid = _project_with_file(client)
    assert not helper.prompts and not qa_fs.list_qa_pairs(pid)           # upload never mines
    src = client.get(f"/api/projects/{pid}/data-prep/sources").json()["sources"][0]
    assert src["parse_status"] == "ready" and src["mining"]["status"] == "parsed"


def test_mine_job_runs_inventory_mine_check_fix_and_stores_every_round(env) -> None:
    client, helper = env
    pid, sid = _project_with_file(client)
    job = _mine(client, pid, sid)
    assert job["status"] == "done", job
    assert job["helper"]["model"] == "fake-model" and job["helper"]["prompt_version"].startswith("mining-")
    assert job["stats"]["fix_rounds"] == 1 and job["stats"]["open"] == 0

    items = db.mining.list_items(pid, sid)
    origins = {it["origin"] for it in items}
    assert origins == {"rule", "agent", "agent_file"}
    texts = " ".join(it["text"] + " " + " ".join(it["values"]) for it in items)
    assert "99 days" not in texts                       # ungrounded agent value dropped
    assert not [it for it in items if it["origin"] == "agent" and it["values"] == ["RC-01", "38.50"]]  # redundant with a rule row
    assert not [it for it in items if it["text"] == "Lane | Rate EUR"]   # a header row is not a data point

    kinds = [r["kind"] for r in db.mining.list_rounds(pid, sid)]
    assert kinds.count("inventory") == 1 and kinds.count("inventory_file") == 1 and "mine" in kinds and "fix" in kinds
    fix = next(r for r in db.mining.list_rounds(pid, sid) if r["kind"] == "fix")
    full = client.get(f"/api/projects/{pid}/mining/rounds/{fix['id']}").json()
    assert "RC-02" in full["messages"][-1]["content"] and "41.75" in full["reply"] and full["accepted"] == 1

    pairs = qa_fs.list_qa_pairs(pid, source_id=sid)
    assert {p["status"] for p in pairs} == {"pending"}                   # never auto-approved
    assert {p["origin"] for p in pairs} == {"mine", "fix"} and all(p["job_id"] == job["id"] for p in pairs)
    checklist = client.get(f"/api/projects/{pid}/mining/files/{sid}/items").json()
    assert checklist["coverage"]["status"] == "review" and checklist["coverage"]["counts"]["open"] == 0
    assert all(it["state"] == "pending" and it["pairs"] for it in checklist["items"])


def test_review_then_reject_reopens_and_fill_sees_the_reason(env) -> None:
    client, helper = env
    pid, sid = _project_with_file(client)
    _mine(client, pid, sid)
    pairs = qa_fs.list_qa_pairs(pid, source_id=sid)
    for p in pairs:
        client.patch(f"/api/projects/{pid}/data-prep/qa/{p['id']}", json={"status": "approved"})
    assert client.get(f"/api/projects/{pid}/mining/files/{sid}/items").json()["coverage"]["status"] == "complete"

    rc02 = next(p for p in pairs if "RC-02" in p["question"])
    covers = client.get(f"/api/projects/{pid}/mining/pairs/{rc02['id']}/items").json()["items"]
    assert covers and all(c["last_cover"] for c in covers)              # rejecting it would leave the row stated by nothing
    client.patch(f"/api/projects/{pid}/data-prep/qa/{rc02['id']}", json={"status": "rejected", "note": "wrong lane name"})
    cov = client.get(f"/api/projects/{pid}/mining/files/{sid}/items").json()["coverage"]
    assert cov["status"] == "gaps" and cov["counts"]["open"] == 1 and not cov["training_ready"]

    helper.prompts.clear()
    job = _mine(client, pid, sid, kind="fill")
    assert job["kind"] == "fill" and job["status"] in ("done", "gaps")
    assert any("wrong lane name" in p for p in helper.prompts)            # the reviewer's reason reached the helper
    assert not any(p.startswith("Document:") and "Checklist of data points" in p for p in helper.prompts)  # no re-mine


def test_exclude_marks_an_item_decided_and_include_reopens_it(env) -> None:
    client, _ = env
    pid, sid = _project_with_file(client)
    _mine(client, pid, sid)
    item = db.mining.list_items(pid, sid)[0]
    r = client.post(f"/api/projects/{pid}/mining/items/{item['id']}/exclude", json={"reason": "boilerplate"})
    assert r.status_code == 200 and r.json()["item"]["excluded_reason"] == "boilerplate"
    states = {i["id"]: i["state"] for i in client.get(f"/api/projects/{pid}/mining/files/{sid}/items").json()["items"]}
    assert states[item["id"]] == "excluded"
    client.post(f"/api/projects/{pid}/mining/items/{item['id']}/exclude", json={"reason": ""})
    states = {i["id"]: i["state"] for i in client.get(f"/api/projects/{pid}/mining/files/{sid}/items").json()["items"]}
    assert states[item["id"]] == "pending"
    assert client.post(f"/api/projects/{pid}/mining/items/nope/exclude", json={"reason": "x"}).status_code == 404


def test_requests_the_state_refuses_are_409_and_nothing_is_queued(env, monkeypatch) -> None:
    client, _ = env
    pid, sid = _project_with_file(client)
    r = client.post(f"/api/projects/{pid}/mining/jobs", json={"source_ids": [sid], "kind": "fill"})
    assert r.status_code == 409 and "not mined yet" in r.json()["detail"]
    _mine(client, pid, sid)
    r = client.post(f"/api/projects/{pid}/mining/jobs", json={"source_ids": [sid], "kind": "mine"})
    assert r.status_code == 409 and "already mined" in r.json()["detail"]
    assert client.post(f"/api/projects/{pid}/mining/jobs", json={"source_ids": ["nope"]}).status_code == 404
    assert client.post(f"/api/projects/{pid}/mining/jobs", json={"source_ids": [sid], "kind": "x"}).status_code == 400
    monkeypatch.setattr("finetune_studio.data.prep.generator.resolve_generator", lambda: None)
    pid2, sid2 = _project_with_file(client, name="other.txt")
    r = client.post(f"/api/projects/{pid2}/mining/jobs", json={"source_ids": [sid2]})
    assert r.status_code == 409 and not db.mining.list_jobs(pid2)
    assert client.post("/api/projects/nope/mining/jobs", json={"source_ids": [sid]}).status_code == 404


def test_reset_drops_inventory_rounds_and_unreviewed_pairs_only(env) -> None:
    client, _ = env
    pid, sid = _project_with_file(client)
    _mine(client, pid, sid)
    keep = qa_fs.list_qa_pairs(pid, source_id=sid)[0]
    client.patch(f"/api/projects/{pid}/data-prep/qa/{keep['id']}", json={"status": "approved"})
    r = client.post(f"/api/projects/{pid}/mining/files/{sid}/reset").json()
    assert r["items_removed"] > 0 and r["pending_pairs_removed"] > 0
    assert [p["id"] for p in qa_fs.list_qa_pairs(pid, source_id=sid)] == [keep["id"]]
    assert not db.mining.list_items(pid, sid) and not db.mining.list_rounds(pid, sid)
    assert _mine(client, pid, sid)["status"] in ("done", "gaps")         # mines again from scratch


def test_failing_helper_fails_the_job_with_the_reason(env) -> None:
    client, helper = env
    helper.fail = True
    pid, sid = _project_with_file(client)
    job = _mine(client, pid, sid)
    assert job["status"] == "failed" and "helper is down" in job["error"]
    assert all(r["error"] for r in db.mining.list_rounds(pid, sid))


def test_calls_never_exceed_the_helpers_concurrency(env) -> None:
    client, helper = env
    helper.delay = 0.3
    pid = client.post("/api/projects", json={"name": f"par-{uuid.uuid4().hex[:6]}", "base_model": "x"}).json()["id"]
    sids = []
    for i in range(4):
        sids.append(_upload(client, pid, f"f{i}.txt", DOC + f"\nFile {i}.\n"))
    assert client.post(f"/api/projects/{pid}/mining/jobs", json={"source_ids": sids}).status_code == 200
    jobs = _wait(pid)
    assert {j["status"] for j in jobs} == {"done"}
    assert helper.max_active == 3                                         # 4 files in parallel, 3 calls at most


def test_export_is_blocked_until_every_file_is_complete(env) -> None:
    client, _ = env
    pid, sid = _project_with_file(client)
    _mine(client, pid, sid)
    r = client.get(f"/api/projects/{pid}/data-prep/export")
    assert r.status_code == 409 and r.json()["incomplete_files"][0]["status"] == "review"
    for p in qa_fs.list_qa_pairs(pid, source_id=sid):
        client.patch(f"/api/projects/{pid}/data-prep/qa/{p['id']}", json={"status": "approved"})
    r = client.get(f"/api/projects/{pid}/data-prep/export")
    assert r.status_code == 200 and int(r.headers["X-Rows"]) >= 4


def test_restart_marks_running_jobs_interrupted(env) -> None:
    from finetune_studio.data.mining import interrupt_after_restart
    client, _ = env
    pid, sid = _project_with_file(client)
    job = db.mining.create_job(pid, sid, "leave.txt", "mine")
    db.mining.update_job(job["id"], status="running")
    assert interrupt_after_restart() >= 1
    assert db.mining.get_job(job["id"])["status"] == "interrupted"


def test_purging_a_file_drops_its_knowledge_but_trash_keeps_it(env) -> None:
    from finetune_studio.knowledge.sources import get_source

    client, _ = env
    pid, sid = _project_with_file(client)
    _mine(client, pid, sid)
    fid = get_source(pid, sid).file_id
    assert client.delete(f"/api/projects/{pid}/files/{fid}").status_code == 200       # trash: restorable
    assert db.mining.list_items(pid, sid) and qa_fs.list_qa_pairs(pid, source_id=sid)
    assert client.post(f"/api/projects/{pid}/files/{fid}/purge").status_code == 200
    assert not db.mining.list_items(pid, sid) and not db.mining.list_rounds(pid, sid)
    assert not qa_fs.list_qa_pairs(pid, source_id=sid) and get_source(pid, sid, include_deleted=True) is None


# ── units ───────────────────────────────────────────────────────────────────────────────────────────────────────────

def test_rule_items_skip_header_rows_and_carry_the_header_into_later_chunks() -> None:
    items = inv.rule_items(["Rates\nLane | Rate\nRC-01 | 38.50", "RC-02 | 41.75\nPlain text without any token at all."])
    rows = [(it.chunk_idx, it.text, it.header) for it in items]
    assert rows == [(1, "RC-01 | 38.50", "Lane | Rate"), (2, "RC-02 | 41.75", "Lane | Rate")]


def test_parse_inventory_is_tolerant_and_ground_drops_invented_values() -> None:
    raw = 'Here you go:\n```json\n[{"fact": "A", "values": ["21 working days", "not in text"]}, {"fact": "", "values": ["x"]}, 3]\n```'
    parsed = inv.parse_inventory(raw)
    assert parsed == [{"fact": "A", "values": ["21 working days", "not in text"]}]
    assert inv.ground(parsed, "Leave is 21  Working Days.") == [{"fact": "A", "values": ["21 working days"]}]
    assert inv.parse_inventory("no json") == []


def test_value_items_need_one_answer_with_every_value_and_rejected_pairs_never_count() -> None:
    items = [{"id": "i1", "origin": "agent", "text": "x", "values": ["garden leave", "in full"], "excluded_reason": ""},
             {"id": "i2", "origin": "agent", "text": "y", "values": ["RC-09"], "excluded_reason": "not a fact"}]
    pairs = [{"id": "p1", "status": "approved", "question": "q", "answer": "Garden leave applies."},
             {"id": "p2", "status": "rejected", "question": "q", "answer": "During garden leave pay continues in full."}]
    states = {s.item_id: s.state for s in item_states(items, pairs)}
    assert states == {"i1": "open", "i2": "excluded"}
    pairs[1]["status"] = "pending"
    assert item_states(items, pairs)[0].state == "pending"
    assert file_status(item_states(items, pairs), has_inventory=True, mining=False) == "review"


def test_cached_coverage_follows_every_pair_and_item_change() -> None:
    from finetune_studio.data.mining.coverage import cached_item_states
    items = [{"id": "i1", "origin": "agent", "text": "x", "values": ["garden leave", "in full"], "excluded_reason": ""}]
    pairs = [{"id": "p1", "status": "pending", "question": "q", "answer": "During garden leave pay continues in full."}]
    assert cached_item_states(items, pairs)[0].state == "pending"
    pairs[0]["status"] = "approved"
    assert cached_item_states(items, pairs)[0].state == "approved"
    pairs[0]["answer"] = "Garden leave applies."
    assert cached_item_states(items, pairs)[0].state == "open"
    items[0]["excluded_reason"] = "not a fact"
    assert cached_item_states(items, pairs)[0].state == "excluded"
    cached_item_states(items, pairs)[0].pair_ids.append("mutated")   # a caller's edit never reaches the cache
    assert cached_item_states(items, pairs)[0].pair_ids == []


def test_statement_coverage_ignores_unrelated_answers_that_share_a_number() -> None:
    s = st.split_statements("Annual leave is 21 working days for employees with 0-2 years of service.")[0]
    unrelated = [st.pair_tokens("p", "How many pallets fit?", "21 pallets and 0-2 spares fit.")]
    related = [st.pair_tokens("q", "How much annual leave for 0-2 years of service?", "21 working days for 0-2 years.")]
    assert st.covering_pairs(s.text, s.tokens, unrelated) == []
    assert st.covering_pairs(s.text, s.tokens, related) == ["q"]


def test_rule_items_need_hard_data_and_ignore_chat_stamps_and_plain_capitals() -> None:
    assert st.hard_tokens("During notice the Company may grant garden leave.") == frozenset()
    assert st.hard_tokens("[2024-09-19 13:10] gunnar: lunch, who wants pierogi") == frozenset()
    assert st.hard_tokens("[2024-09-19 09:18] jarek: cut-off is now THURSDAY 15:00") == {"thursday", "15:00"}
    assert st.hard_tokens("Lane RC-01 costs EUR 38.50 per day") == {"rc-01", "eur", "38.5"}
    items = inv.rule_items(["The Effective Date is defined below.\nThe fee is EUR 120,000 per container."])
    assert [it.text for it in items] == ["The fee is EUR 120,000 per container."]


def test_a_combined_fact_may_be_stated_across_atomic_pairs() -> None:
    item = {"id": "i", "origin": "agent_file", "text": "x", "excluded_reason": "",
            "values": ["EUR 57.5m", "EUR 21.0m", "2.74x"]}
    pairs = [{"id": "a", "status": "approved", "question": "q", "answer": "Net debt was EUR 57.5m against EUR 21.0m LTM EBITDA."},
             {"id": "b", "status": "approved", "question": "q", "answer": "Leverage was 2.74x, with net debt of EUR 57.5m."},
             {"id": "c", "status": "approved", "question": "q", "answer": "Headroom was 2.74x the minimum."}]   # one value only: not about it
    state = item_states([item], pairs)[0]
    assert state.state == "approved" and sorted(state.pair_ids) == ["a", "b"]
    assert item_states([item], pairs[:1])[0].state == "open"


def test_a_fix_round_skip_is_a_proposal_until_a_reviewer_confirms_it(env) -> None:
    client, helper = env
    pid, sid = _project_with_file(client, DOC + "\nThe page reference PR-77 is printed at the foot of every page.\n")
    job = _mine(client, pid, sid)
    assert job["status"] == "gaps" and job["stats"]["proposed"] == 1 and "proposed for exclusion" in job["message"]
    fixes = [r for r in db.mining.list_rounds(pid, sid) if r["kind"] == "fix"]
    assert len(fixes) == 1                                  # the proposed item leaves the next fix round: nothing left to ask

    checklist = client.get(f"/api/projects/{pid}/mining/files/{sid}/items").json()
    prop = [it for it in checklist["items"] if it["proposed_reason"]]
    assert [(it["state"], it["proposed_reason"]) for it in prop] == [("open", "page footer")]
    assert checklist["coverage"]["counts"]["proposed"] == 1 and checklist["coverage"]["status"] == "gaps"   # never auto-excluded

    assert client.post(f"/api/projects/{pid}/mining/files/{sid}/proposals/confirm", json={"item_ids": "x"}).status_code == 422
    r = client.post(f"/api/projects/{pid}/mining/files/{sid}/proposals/confirm", json={})
    assert r.status_code == 200 and r.json()["confirmed"] == 1
    item = db.mining.get_item(pid, prop[0]["id"])
    assert item["excluded_reason"] == "helper: page footer"
    cov = client.get(f"/api/projects/{pid}/mining/files/{sid}/items").json()["coverage"]
    assert cov["status"] == "review" and cov["counts"]["excluded"] == 1

    client.post(f"/api/projects/{pid}/mining/items/{item['id']}/exclude", json={"reason": ""})    # reviewer disagrees
    again = db.mining.get_item(pid, item["id"])
    assert again["excluded_reason"] == "" and again["proposed_reason"] == ""


def test_parse_skips_needs_a_checklist_number_and_a_reason() -> None:
    raw = '[{"q": "Q?", "a": "A."}, {"skip": 2, "reason": "heading"}, {"skip": 9, "reason": "x"}, {"skip": 1}, {"skip": "z"}]'
    assert inv.parse_skips(raw, 3) == {1: "heading"}
    assert inv.parse_skips("not json", 3) == {}


def test_clause_numbers_are_not_data_and_large_numbers_are_one_token() -> None:
    assert st.hard_tokens("3.5 Up to 6 days of leave carry over") == frozenset()
    assert st.hard_tokens('5.2 In clause 9.1(a) "EUR 4,000,000" is replaced by "EUR 6,000,000".') == {"9.1", "eur", "4000000", "6000000"}
    assert st.hard_tokens("2024 revenue was EUR 29.5m") == {"2024", "eur", "29.5m"}
    assert st.hard_tokens("Table 3 100 units") == {"100"}               # a plain space never joins numbers
    assert st.hard_tokens("E-mail notices; Nordhafen Bank AG") == frozenset()   # sentence-initial capital / 2 letters: no code
    assert "6000000" in st.pair_tokens("p", "q", "The cap is now EUR 6,000,000.").answer
    assert "57.5" in st.pair_tokens("p", "q", "Net debt is EUR 57.5m.").answer


def test_a_code_with_a_one_digit_version_is_one_token() -> None:
    """Core mine 2026-10-21: "KC-Tag 4" lost its "4" (too short for a token), so the answer "KC-Tag 3 end of life"
    stated the item "The KC-Tag 4 is the successor of the KC-Tag 3"."""
    item = "The KC-Tag 4 is the successor of the KC-Tag 3 and works with the KC-Hub 200 gateway."
    assert {"kc-tag4", "kc-tag3", "kc-hub", "200"} <= st.hard_tokens(item)
    wrong = st.pair_tokens("w", "Which related items are listed?", "The KC-Hub 200 gateway and KC-Tag 3 end of life.")
    right = st.pair_tokens("r", "What is the KC-Tag 4 the successor of?",
                           "The KC-Tag 4 is the successor of the KC-Tag 3 and works with the KC-Hub 200 gateway.")
    assert st.covering_pairs(item, st.hard_tokens(item), [wrong]) == []
    assert st.covering_pairs(item, st.hard_tokens(item), [right]) == ["r"]
    assert "kc-tag4" in st.pair_tokens("p", "q", "The KC-Tag4 runs 5 years.").answer    # written without the space
    assert st.hard_tokens("Level 4 Black") == frozenset()                             # not a code
    assert st.hard_tokens("KC-Tag 4.5 firmware") == {"kc-tag", "4.5"}                 # a decimal is no version


def test_wrapped_lines_are_one_statement() -> None:
    text = "Up to 6 days of unused leave may be carried over into the following\nyear, but must be taken by 31 March.\n" \
           "Gross misconduct includes falsification of a record (code GM-1)\n(GM-2) and theft (GM-3).\nNext Heading"
    assert [s.text for s in st.split_statements(text)] == [
        "Up to 6 days of unused leave may be carried over into the following year, but must be taken by 31 March.",
        "Gross misconduct includes falsification of a record (code GM-1) (GM-2) and theft (GM-3).",
    ]


def test_a_rule_item_without_checkable_data_is_no_data_point() -> None:
    """Inventories mined under an older tokenizer keep e.g. a clause number as their only token: they leave the checklist."""
    items = [{"id": "old", "origin": "rule", "text": "3.5 Up to 6 days of leave", "values": ["3.5"], "excluded_reason": ""},
             {"id": "row", "origin": "rule", "text": "RC-01 | 38.50", "values": ["rc-01", "38.5"], "excluded_reason": ""}]
    assert [s.item_id for s in item_states(items, [])] == ["row"]


def test_a_pair_stating_one_value_counts_when_it_is_about_the_item() -> None:
    item = {"id": "i", "origin": "agent_file", "excluded_reason": "",
            "text": "The electrician visit for the C-36 conversion moved from Wed 02.10 to Thu 03.10.",
            "values": ["C-36", "Wed 02.10", "Thu 03.10"]}
    about = [{"id": "a", "status": "approved", "question": "q", "answer": "The electrician visit for C-36 was pencilled for Wed 02.10."},
             {"id": "b", "status": "approved", "question": "q", "answer": "The electrician visit moved to Thu 03.10."}]
    stray = [about[0], {"id": "c", "status": "approved", "question": "q", "answer": "Payroll runs on Thu 03.10."}]
    assert item_states([item], about)[0].state == "approved"
    assert item_states([item], stray)[0].state == "open"          # shares a date, not the subject


def test_ground_drops_structure_facts_and_phrase_values() -> None:
    text = "3. Rollback. Section 2 is titled no ping power. Do not use the old rate of EUR 0.29 per km because we wanted to beat the traffic every single day."
    entries = [
        {"fact": "Section 2 is titled 'no ping power'.", "values": ["no ping power"]},
        {"fact": "Old rate is not used.", "values": ["EUR 0.29 per km", "we wanted to beat the traffic every single day"]},
        {"fact": "Only a phrase.", "values": ["we wanted to beat the traffic every single day"]},
    ]
    assert inv.ground(entries, text) == [{"fact": "Old rate is not used.", "values": ["EUR 0.29 per km"]}]


def test_table_value_with_unit_from_the_header_is_grounded() -> None:
    from finetune_studio.data.prep.qa_validate import validate_qa_batch
    chunk = ("Net debt and covenants at 30 September 2024\nItem | EUR m\nItem: Term loan (Nordhafen Bank AG) | EUR m: 38.5\n"
             "Item: Cash and equivalents | EUR m: -9.7\n")
    ok = validate_qa_batch([{"q": "What are the cash and equivalents in the Board Pack net debt table?",
                             "a": "Cash and equivalents are EUR -9.7m."}], chunk, seen_questions=set())
    assert len(ok.accepted) == 1
    bad = validate_qa_batch([{"q": "What are the cash and equivalents in the Board Pack net debt table?",
                              "a": "Cash and equivalents are EUR -9.8m."}], chunk, seen_questions=set())
    assert not bad.accepted


def test_document_identifiers_ground_later_chunk_pairs() -> None:
    from finetune_studio.data.mining.job import document_identifiers
    from finetune_studio.data.prep.qa_validate import validate_qa_batch
    ids = document_identifiers("Runbook RB-017: KC-Hub 200 offline", "Owner: IT Support.")
    chunk = "Firmware: the hub has A/B slots."
    pair = [{"q": "What firmware slots does the hub in the KC-Hub offline runbook have?",
             "a": "According to RB-017, the KC-Hub 200 hub has A/B firmware slots."}]
    assert not validate_qa_batch(pair, chunk, seen_questions=set()).accepted
    assert len(validate_qa_batch(pair, f"{chunk}\n{ids}", seen_questions=set()).accepted) == 1


def test_whole_file_item_is_mined_against_every_chunk_holding_its_values() -> None:
    from finetune_studio.data.mining import job as mjob
    j = object.__new__(mjob.MiningJob)
    texts = ["CHG-2291 set the batch window to 6000000 ms.", "Weather was fine.", "CHG-2294 corrected it to 6000 ms."]
    j.chunks = [mjob.Chunk(i + 1, t, t, t) for i, t in enumerate(texts)]
    items = [{"id": "a", "chunk_idx": 2, "origin": "agent", "values": ["fine"]},
             {"id": "f", "chunk_idx": 1, "origin": "agent_file", "values": ["CHG-2291", "6000000 ms", "CHG-2294"]}]
    units = j._units(items)
    assert [(u[0].idx, [i["id"] for i in u[1]]) for u in units] == [(2, ["a"]), (1, ["f"])]
    span = units[1][0]
    assert span.span and "CHG-2291" in span.grounding and "CHG-2294" in span.grounding and "Weather" not in span.grounding


def test_whole_file_items_share_one_joined_passage_when_it_fits() -> None:
    from finetune_studio.data.mining import job as mjob
    j = object.__new__(mjob.MiningJob)
    texts = ["CHG-2291 set 6000000 ms.", "Filler.", "CHG-2294 fixed it to 6000 ms.", "Owner Priya Nair.", "Priya left in May."]
    j.chunks = [mjob.Chunk(i + 1, t, t, t) for i, t in enumerate(texts)]
    items = [{"id": "f1", "chunk_idx": 1, "origin": "agent_file", "values": ["CHG-2291", "CHG-2294"]},
             {"id": "f2", "chunk_idx": 4, "origin": "agent_file", "values": ["Priya Nair", "May"]}]
    units = j._units(items)
    assert len(units) == 1 and [i["id"] for i in units[0][1]] == ["f1", "f2"]
    assert "Filler" not in units[0][0].grounding and units[0][0].span
    assert mjob._values_of(units[0][1], units[0][0]) == ["CHG-2291", "CHG-2294", "Priya Nair", "May"]


def test_joined_passages_are_packed_up_to_the_prompt_budget() -> None:
    from finetune_studio.data.mining import job as mjob
    j = object.__new__(mjob.MiningJob)
    j.chunks = [mjob.Chunk(i + 1, f"V{i + 1} " + "x" * 1997, f"V{i + 1} " + "x" * 1997, "p" * 2000) for i in range(6)]
    items = [{"id": f"f{k}", "chunk_idx": a, "origin": "agent_file", "values": [f"V{a}", f"V{b}"]}
             for k, (a, b) in enumerate([(1, 2), (2, 3), (4, 5), (5, 6)])]
    # chunks 1-3 = 6000 chars fit; adding 4-5 would be 10000 > 7500, so a second passage holds chunks 4-6
    got = [([i["id"] for i in todo], ch.grounding.count("V")) for ch, todo in j._units(items)]
    assert got == [(["f0", "f1"], 3), (["f2", "f3"], 3)]


def test_one_malformed_inventory_object_does_not_drop_the_rest() -> None:
    # a real helper reply (2026-10-14): a stray '",' inside object 4 made the whole array unparseable: 0 items kept
    raw = ('[\n {"fact": "Form KC-QF-140.", "values": ["KC-QF-140"]},\n {"fact": "Issue at +9.4 C.", ",\n'
           '   "values": ["+9.4 C"]},\n {"fact": "Closed by T. Kowarzik.", "values": ["T. Kowarzik"]},\n'
           ' {"fact": "Root cause RF-88.", "values": ["RF-88"]}\n]')
    got = [e["values"][0] for e in inv.parse_inventory(raw)]
    assert got == ["KC-QF-140", "T. Kowarzik", "RF-88"]


def test_pair_salvage_survives_a_stray_quote() -> None:
    from finetune_studio.data.prep.parsers import parse_qa_json
    raw = ('[{"q": "Who raised KC-QF-140?", "a": "Priya Nair raised it."}, {"q": "Issue?", ", "a": "x"},'
           ' {"q": "Who closed KC-QF-140?", "a": "T. Kowarzik closed it."}]')
    qs = [p["q"] for p in parse_qa_json(raw, 10)]
    assert "Who raised KC-QF-140?" in qs and "Who closed KC-QF-140?" in qs


def test_a_name_after_an_initial_is_a_name() -> None:
    from finetune_studio.data.mining.statements import drop_redundant
    from finetune_studio.data.prep.tokens import distinctive_tokens
    assert "kowarzik" in distinctive_tokens("The form KC-QF-140 was closed by T. Kowarzik.")
    assert "aaltonen" in distinctive_tokens("Call Dr. Aaltonen first.")
    assert "fleet" in distinctive_tokens("It ended. The next day Fleet called.")
    kept = [{"q": "Form?", "a": "The form is KC-QF-140."}, {"q": "Raised?", "a": "KC-QF-140 was raised on 3 October."}]
    new = [{"q": "Who closed KC-QF-140?", "a": "The form KC-QF-140 was closed by T. Kowarzik."}]
    assert drop_redundant(new, kept) == new        # was dropped: 'Kowarzik' read as a sentence start


def test_mine_and_fix_prompts_keep_comparison_direction() -> None:
    """RESULTS §16: 'beat the budget by 1.3m' was mined as 'missed'; both stages must forbid flipping."""
    from finetune_studio.data.mining import prompts

    mine = prompts.mine_messages("Board pack", "Q3 revenue beat the budget by 1.3m.", ["revenue vs budget"])
    fix = prompts.fix_messages("Board pack", "Q3 revenue beat the budget by 1.3m.", ["revenue vs budget"],
                               rejected=[("Did Q3 revenue meet the budget?", "It missed it.", "direction flipped")])
    for msgs in (mine, fix):
        system = msgs[0]["content"]
        assert "Keep the direction of every comparison exactly as written" in system
        assert "beat/missed" in system and "Never turn one into the other" in system
    assert prompts.PROMPT_VERSION == "mining-v5"
