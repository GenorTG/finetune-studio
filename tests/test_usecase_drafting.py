"""Use-case drafting: raw texts -> assistant drafts -> review -> build from approved rows only."""
from __future__ import annotations

import json
import re

import pytest
from fastapi.testclient import TestClient

from finetune_studio.usecases import drafting

LABELS = ["billing", "shipping", "account"]


# ── parsing the upload ───────────────────────────────────────────────────────

def test_texts_come_from_json_jsonl_and_plain_text() -> None:
    assert drafting.parse_texts(b'["a one", {"text": "b two"}, {"input": "c three"}, 5, ""]') == ["a one", "b two",
                                                                                                    "c three"]
    assert drafting.parse_texts(b'{"prompt": "x"}\n\n{"body": "y"}\n', "t.jsonl") == ["x", "y"]
    assert drafting.parse_texts(b"first email\nsecond line\n\nsecond email\n") == ["first email\nsecond line",
                                                                                    "second email"]
    assert drafting.parse_texts(b"one\ntwo\nOne\n") == ["one", "two"]          # one per line; duplicates merged


@pytest.mark.parametrize("raw,msg", [(b"", "empty"), (b"[1, 2]", "no text"), (b"{bad", "line 1"),
                                     (b"\xff\xfe", "UTF-8")])
def test_unusable_uploads_say_why(raw: bytes, msg: str) -> None:
    with pytest.raises(drafting.DraftError, match=msg):
        drafting.parse_texts(raw)


def test_limits_are_enforced() -> None:
    with pytest.raises(drafting.DraftError, match="at most"):
        drafting.parse_texts(json.dumps([f"t{i}" for i in range(drafting.MAX_TEXTS + 1)]).encode())
    with pytest.raises(drafting.DraftError, match="longer than"):
        drafting.parse_texts(json.dumps(["x" * (drafting.MAX_TEXT_CHARS + 1)]).encode())


# ── options, prompts, replies ────────────────────────────────────────────────

@pytest.mark.parametrize("template,opts,msg", [
    ("distillation", drafting.DraftOptions(), "teacher"),
    ("classification", drafting.DraftOptions(labels=["only"]), "two labels"),
    ("extraction", drafting.DraftOptions(), "fields"),
])
def test_options_are_checked(template: str, opts: drafting.DraftOptions, msg: str) -> None:
    with pytest.raises(drafting.DraftError, match=msg):
        drafting.check_options(template, opts)


def test_classification_replies_map_onto_the_label_list() -> None:
    opts = drafting.DraftOptions(labels=LABELS)
    assert drafting.parse_reply("classification", "Billing.", opts) == ("billing", "")
    assert drafting.parse_reply("classification", "The label is: shipping", opts) == ("shipping", "")
    draft, why = drafting.parse_reply("classification", "refunds", opts)
    assert draft == "" and "not one of the labels" in why
    assert drafting.parse_reply("classification", "billing or shipping", opts)[1]   # ambiguous: not guessed


def test_extraction_replies_are_parsed_and_checked_against_the_schema() -> None:
    schema = {"type": "object", "properties": {"total": {"type": "number"}}, "required": ["total"]}
    opts = drafting.DraftOptions(schema=schema)
    assert drafting.parse_reply("extraction", '```json\n{"total": 12.5}\n```', opts) == ({"total": 12.5}, "")
    value, why = drafting.parse_reply("extraction", '{"total": "twelve"}', opts)
    assert value == {"total": "twelve"} and "schema" in why
    assert drafting.parse_reply("extraction", "no json here", opts) == ("", "the reply was not one JSON object")


def test_prompts_carry_the_options() -> None:
    opts = drafting.DraftOptions(labels=LABELS, fields=["vendor", "total"], style_guide="short and warm",
                                 hint="support tickets")
    assert "- billing" in drafting.prompt_for("classification", "T", opts) and "support tickets" in \
        drafting.prompt_for("classification", "T", opts)
    assert "vendor, total" in drafting.prompt_for("extraction", "T", opts)
    assert "short and warm" in drafting.prompt_for("style", "Sure, done.", opts)


def test_draft_rows_keeps_failed_and_unusable_rows_for_the_user() -> None:
    def generate(prompt: str) -> str:
        text = prompt.rsplit("Text:\n", 1)[1]
        if "boom" in text:
            raise RuntimeError("rate limited")
        return "nonsense" if "odd" in text else "billing"

    progress: list[dict] = []
    rows = drafting.draft_rows("classification", ["a charge", "odd one", "boom"], generate,
                               drafting.DraftOptions(labels=LABELS), workers=2, progress=progress.append)
    assert [r["draft"] for r in rows] == ["billing", "", ""]
    assert rows[1]["note"].startswith("the reply was not") and "rate limited" in rows[2]["note"]
    assert all(r["status"] == "pending" for r in rows) and progress[-1] == {"done": 3, "planned": 3, "drafted": 1}


def test_an_assistant_that_never_answers_fails_the_run() -> None:
    def generate(prompt: str) -> str:
        raise RuntimeError("down")

    with pytest.raises(drafting.DraftError, match="failed 5 calls in a row"):
        drafting.draft_rows("classification", [f"t{i}" for i in range(8)], generate,
                            drafting.DraftOptions(labels=LABELS))


def test_style_drafts_are_requests_and_the_text_stays_the_answer() -> None:
    row = {"input": "Yes, you can change it under Orders.", "draft": "Can I change my address?", "status": "approved"}
    assert drafting.example_for("style", row) == {"prompt": "Can I change my address?",
                                                  "answer": "Yes, you can change it under Orders."}


# ── routes: draft -> review -> build ─────────────────────────────────────────

def _project(client: TestClient) -> str:
    r = client.post("/api/projects", json={"name": "drafts", "base_model": "x/test"})
    assert r.status_code == 200, r.text
    return r.json()["id"]


def _fake_assistant(monkeypatch: pytest.MonkeyPatch, reply=None) -> list[str]:
    prompts: list[str] = []

    def generate(prompt: str) -> str:
        prompts.append(prompt)
        if reply is not None:
            return reply(prompt)
        text = prompt.rsplit("Text:\n", 1)[1]
        return next(lab for lab in LABELS if lab in text)

    monkeypatch.setattr("finetune_studio.data.prep.preference.helper_generate", lambda *a, **k: generate)
    return prompts


def _texts(n: int = 30) -> bytes:
    return json.dumps([f"ticket {i} is about {lab} please help {i}" for lab in LABELS for i in range(n)]).encode()


def _draft(client: TestClient, pid: str, raw: bytes, **opts) -> object:
    options = {"template": "classification", "labels": "billing, shipping, account", **opts}
    return client.post(f"/api/projects/{pid}/usecases/drafts", files={"file": ("t.json", raw)},
                       data={"options": json.dumps(options)})


def test_draft_review_and_build_from_approved_rows_only(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    prompts = _fake_assistant(monkeypatch)
    pid = _project(client)
    r = _draft(client, pid, _texts(), name="Ticket drafts", hint="support tickets")
    assert r.status_code == 201, r.text
    ds = r.json()
    assert ds["template"] == "classification" and len(ds["rows"]) == 90 and len(prompts) == 90
    assert all(r["status"] == "pending" and r["draft"] in LABELS for r in ds["rows"])
    did = ds["id"]

    built = client.post(f"/api/projects/{pid}/usecases/drafts/{did}/build", json={})
    assert built.status_code == 422 and built.json()["code"] == "nothing_approved"   # nothing trains on drafts

    rows = ds["rows"]
    changes = [{"id": r["id"], "status": "approved"} for r in rows[:84]]
    changes.append({"id": rows[84]["id"], "status": "rejected"})
    changes.append({"id": rows[85]["id"], "draft": "account", "set_draft": True, "status": "approved"})
    upd = client.patch(f"/api/projects/{pid}/usecases/drafts/{did}/rows", json={"changes": changes})
    assert upd.status_code == 200, upd.text
    assert upd.json()["counts"] == {"pending": 4, "approved": 85, "rejected": 1, "empty": 0}
    assert next(r for r in upd.json()["rows"] if r["id"] == rows[85]["id"])["draft"] == "account"

    m = client.post(f"/api/projects/{pid}/usecases/drafts/{did}/build", json={"name": "From drafts"})
    assert m.status_code == 201, m.text
    assert m.json()["examples"] == 85 and m.json()["name"] == "From drafts"
    listed = client.get(f"/api/projects/{pid}/usecases/drafts").json()["drafts"]
    assert listed[0]["id"] == did and listed[0]["counts"]["approved"] == 85
    assert client.delete(f"/api/projects/{pid}/usecases/drafts/{did}").json() == {"ok": True}
    assert client.get(f"/api/projects/{pid}/usecases/drafts/{did}").status_code == 404


def test_row_edits_are_checked_and_an_empty_draft_cannot_be_approved(client: TestClient,
                                                                    monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_assistant(monkeypatch, reply=lambda p: "billing" if "0" in p.rsplit("Text:", 1)[1] else "no idea")
    pid = _project(client)
    ds = _draft(client, pid, b'["ticket 0", "ticket 1"]').json()
    good, empty = ds["rows"]
    assert empty["draft"] == "" and empty["note"]
    url = f"/api/projects/{pid}/usecases/drafts/{ds['id']}/rows"
    r = client.patch(url, json={"changes": [{"id": empty["id"], "status": "approved"}]})
    assert r.status_code == 400 and "usable draft" in r.json()["detail"]
    r = client.patch(url, json={"changes": [{"id": empty["id"], "draft": "refunds", "set_draft": True}]})
    assert r.status_code == 400 and "not one of the labels" in r.json()["detail"]
    r = client.patch(url, json={"changes": [{"id": "r9999", "status": "approved"}]})
    assert r.status_code == 404
    # All or nothing: a bad change leaves the good one unsaved too.
    r = client.patch(url, json={"changes": [{"id": good["id"], "status": "approved"},
                                            {"id": empty["id"], "status": "approved"}]})
    assert r.status_code == 400
    assert client.get(f"/api/projects/{pid}/usecases/drafts/{ds['id']}").json()["counts"]["approved"] == 0


def test_bad_requests_are_refused_before_any_assistant_call(client: TestClient,
                                                           monkeypatch: pytest.MonkeyPatch) -> None:
    prompts = _fake_assistant(monkeypatch)
    pid = _project(client)
    assert _draft(client, pid, _texts(), labels="billing").status_code == 400
    assert _draft(client, pid, _texts(), template="distillation").status_code == 400
    assert _draft(client, pid, b"").status_code == 400
    assert client.post(f"/api/projects/{pid}/usecases/drafts", files={"file": ("t.json", b'["x"]')},
                       data={"options": "[1]"}).status_code == 400
    assert prompts == []
    assert _draft(client, "nope0000", _texts()).status_code == 404


def test_no_usable_assistant_is_a_409_with_the_reason(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("finetune_studio.data.prep.preference.helper_generate", lambda *a, **k: None)
    monkeypatch.setattr("finetune_studio.data.prep.generator.helper_resolution_error",
                        lambda *a, **k: "Load the assistant first.")
    pid = _project(client)
    r = _draft(client, pid, _texts(2))
    assert r.status_code == 409 and r.json()["code"] == "no_model" and "Load the assistant" in r.json()["detail"]
    assert client.get(f"/api/projects/{pid}/usecases/drafts/progress").json()["running"] is False


def test_extraction_and_style_drafts_build_into_their_templates(client: TestClient,
                                                                monkeypatch: pytest.MonkeyPatch) -> None:
    def reply(prompt: str) -> str:
        if "JSON object" in prompt:
            n = re.search(r"invoice (\d+)", prompt).group(1)
            return json.dumps({"invoice_no": n, "total": float(n)})
        return "How do I do what this says: " + prompt.rsplit("Reply:\n", 1)[1] + "?"

    _fake_assistant(monkeypatch, reply=reply)
    pid = _project(client)
    raw = json.dumps([f"invoice {i} from Acme, total {i}" for i in range(70)]).encode()
    ds = _draft(client, pid, raw, template="extraction", labels="", fields="invoice_no, total").json()
    assert ds["rows"][3]["draft"] == {"invoice_no": "3", "total": 3.0}
    url = f"/api/projects/{pid}/usecases/drafts/{ds['id']}"
    client.patch(url + "/rows", json={"changes": [{"id": r["id"], "status": "approved"} for r in ds["rows"]]})
    m = client.post(url + "/build", json={})
    assert m.status_code == 201, m.text
    assert m.json()["template"] == "extraction"

    raw = json.dumps([f"Sure, reply number {i}: open Orders and pick change." for i in range(70)]).encode()
    ds = _draft(client, pid, raw, template="style", labels="", style_guide="short").json()
    assert ds["rows"][0]["draft"].startswith("How do I")
    url = f"/api/projects/{pid}/usecases/drafts/{ds['id']}"
    client.patch(url + "/rows", json={"changes": [{"id": r["id"], "status": "approved"} for r in ds["rows"]]})
    assert client.post(url + "/build", json={}).status_code == 201


def test_drafting_is_an_assistant_job() -> None:
    from finetune_studio.models import assistant
    assert "drafting" in assistant.JOB_IDS and assistant.job("drafting").default == assistant.ASSISTANT


def test_the_card_drives_every_drafting_endpoint() -> None:
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "src" / "finetune_studio" / "webui"
    html = (root / "templates" / "project_training.html").read_text(encoding="utf-8")
    js = (root / "static" / "js" / "usecases.js").read_text(encoding="utf-8")
    for el in ("uc-draft", "uc-draft-file", "uc-draft-hint", "uc-draft-fields", "uc-draft-btn", "uc-draft-status",
               "uc-draft-sets", "uc-draft-review"):
        assert f'id="{el}"' in html, el
    for call in ("api.post(`${base}/drafts`, fd)", "`${base}/drafts/progress`", "api.get(`${base}/drafts`)",
                 "api.get(`${base}/drafts/${encodeURIComponent(did)}`)", "/rows`, { changes })",
                 "/build`,", "api.del(`${base}/drafts/"):
        assert call in js, call
    assert "set_draft: true" in js and "Approve all drafted" in js
