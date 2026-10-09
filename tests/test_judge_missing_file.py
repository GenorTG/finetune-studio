"""A local judge whose GGUF was deleted: pickers disable it, and judging it fails with a sentence, not a bare path
(live finding 2026-10-09: 'Judge failed: /…/Qwen3-30B-A3B-Instruct-2507-IQ4_XS.gguf')."""
from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from finetune_studio.testing.judge import (
    JudgeCase,
    JudgeUnavailable,
    list_judge_providers,
    open_judge,
    parse_judge_reply,
)


def _local_provider(pid: str, path: Path) -> str:
    from finetune_studio.models.manager import get_manager

    get_manager().upsert_provider(id=pid, name="Gone judge", kind="local_gguf", model_id=str(path), base_url="", api_key="")
    return pid


def test_a_deleted_gguf_judge_is_flagged_and_refused_with_a_clear_message(client: TestClient, tmp_path: Path) -> None:
    gone = tmp_path / "deleted-judge.gguf"
    pid = _local_provider("gone-judge", gone)
    row = next(r for r in list_judge_providers() if r["id"] == pid)
    assert row["local"] is True and row["file_missing"] is True
    with pytest.raises(JudgeUnavailable) as exc, open_judge(pid):
        pass
    assert "not found on disk" in str(exc.value) and str(gone) in str(exc.value) and "pick another judge" in str(exc.value)


def test_present_gguf_is_not_flagged(client: TestClient, tmp_path: Path) -> None:
    here = tmp_path / "judge.gguf"
    here.write_bytes(b"x")
    pid = _local_provider("here-judge", here)
    assert next(r for r in list_judge_providers() if r["id"] == pid)["file_missing"] is False


def test_pickers_disable_a_missing_judge_file(client: TestClient, tmp_path: Path) -> None:
    _local_provider("gone-judge-2", tmp_path / "deleted-judge.gguf")
    project = client.post("/api/projects", json={"name": "jm", "base_model": "x/test"}).json()["id"]
    for page in (f"/projects/{project}/testing", f"/projects/{project}/compare"):
        html = client.get(page).text
        assert 'value="gone-judge-2" disabled' in html or 'value="gone-judge-2" data-local="1" disabled' in html, page
        assert "file missing on disk" in html


def test_run_detail_never_preselects_a_disabled_judge() -> None:
    html = (Path(__file__).resolve().parents[1] / "src/finetune_studio/webui/templates/project_testing.html").read_text(encoding="utf-8")
    assert "if (opt && !opt.disabled) sel.value = r.judge_provider_id;" in html


def test_a_malformed_checklist_reply_is_an_error_not_the_bare_verdict() -> None:
    """Live case e039 (2026-10-09): the judge's JSON broke inside "evidence", the regex fallback took "pass"
    although the checklist said the only fact was missing. Such a reply is unreadable, so the case is retried / left unjudged."""
    raw = '```json\n{"facts": [{"fact": "5", "status": "missing", "evidence":"}], "extra_wrong": false, "verdict": "pass"}\n```'
    case = JudgeCase(question="How long?", correct_answer="5", model_answer="I don't know from the provided documents.")
    r = parse_judge_reply(raw, case)
    assert r.verdict == "" and "checklist could not be read" in r.error
    clean = parse_judge_reply('{"verdict": "fail", "reasoning": "missing"}', case)
    assert clean.verdict == "fail"
