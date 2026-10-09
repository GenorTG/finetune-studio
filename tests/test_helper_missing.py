"""B1: a missing helper GGUF gives an actionable error and can be re-pointed."""

from __future__ import annotations

from finetune_studio.models.helper import DEFAULT_HELPER_PROVIDER_ID


def test_load_missing_helper_is_actionable(client, tmp_path):
    ghost = tmp_path / "nope.gguf"
    client.post("/api/providers", json={
        "id": DEFAULT_HELPER_PROVIDER_ID, "kind": "local_gguf", "model_id": str(ghost),
    })
    r = client.post(f"/api/providers/{DEFAULT_HELPER_PROVIDER_ID}/load", json={})
    assert r.status_code == 400
    body = r.json()
    assert body["code"] == "helper_missing"
    assert "Model library" in body["error"]
    assert "HFValidationError" not in body["error"]
    st = client.get("/api/providers/helper/status").json()
    assert st["ok"] is False and st["missing_path"] == str(ghost)


def test_helper_use_validates_and_persists(client, tmp_path):
    good = tmp_path / "m.gguf"
    good.write_bytes(b"GGUF")
    assert client.post("/api/providers/helper/use", json={"path": str(tmp_path / "x.gguf")}).status_code == 400
    assert client.post("/api/providers/helper/use", json={"path": str(tmp_path)}).status_code == 400
    r = client.post("/api/providers/helper/use", json={"path": str(good)})
    assert r.status_code == 200
    assert client.get("/api/providers/helper/status").json()["ok"] is True


def test_export_contents_resolves_cwd_relative_run_paths(client, tmp_path, monkeypatch):
    from finetune_studio import db
    from finetune_studio.webui.routes.project_models import _normalize_export_path

    monkeypatch.chdir(tmp_path)
    (tmp_path / "output" / "r1" / "merged").mkdir(parents=True)
    assert _normalize_export_path("output/r1/merged") == str(tmp_path / "output" / "r1" / "merged")
    assert _normalize_export_path("etc/passwd") == "/etc/passwd"
    assert db is not None


def test_run_creation_backfills_project_base_model(client):
    from finetune_studio import db

    p = db.create_project("bf")
    assert not p["base_model"]
    db.create_run(p["id"], "r", base_model="Qwen/Qwen3-0.6B")
    assert db.get_project(p["id"])["base_model"] == "Qwen/Qwen3-0.6B"
    db.create_run(p["id"], "r2", base_model="other/model")
    assert db.get_project(p["id"])["base_model"] == "Qwen/Qwen3-0.6B"


def test_unknown_project_pages_are_404(client):
    for path in ("/projects/nope", "/projects/nope/training", "/projects/nope/benchmarks"):
        r = client.get(path, headers={"accept": "text/html"}, follow_redirects=False)
        assert r.status_code == 404, path


def test_short_run_summary_uses_mean_loss_not_zero():
    from finetune_studio.training.engine import TrainingState, apply_trainer_log

    st = TrainingState()
    apply_trainer_log(st, {"train_loss": 4.665}, global_step=3, epoch=1.0, total_steps=3, elapsed=1.0)
    assert st.final_loss == 4.665
    assert "last loss=0.0" not in st.log_lines[-1]


def test_abstention_is_not_a_pass():
    """A refusal to a question the key answers is the judge's call, not a substring match: the prompt says so and the
    verdict the judge returns is what gets saved, whatever words the answer contains."""
    from finetune_studio import db
    from finetune_studio.testing import judge as judge_mod
    from finetune_studio.testing.judge import (
        JudgeCase,
        LoadedJudge,
        build_judge_messages,
    )
    from finetune_studio.testing.judging import judge_benchmark

    declined = "The provided text does not mention who founded Zorblax Tomasz Wrona."
    system = build_judge_messages(JudgeCase("Who founded Zorblax?", "Tomasz Wrona founded Zorblax in 2019.", declined))[0]
    assert "a refusal" in system["content"] and "I don't know" in system["content"] and '"fail" = no fact present' in system["content"]

    pid = db.create_project(name="p")["id"]
    rid = db.create_run(pid, "run")["id"]
    bid = db.create_benchmark(rid, "quiz", {}, status="running", kind="suite")["id"]
    cid = db.create_case(bid, rid, "c", "g", "Who founded Zorblax?", "Tomasz Wrona founded Zorblax in 2019.", declined, [])
    reply = '{"reasoning": "declined although the key answers it", "verdict": "fail"}'
    judge_benchmark(bid, LoadedJudge(chat=lambda _m: reply, provider_id="f", model="m", label="F", concurrent=False))
    case = db.get_case(cid)
    assert case["verdict"] == "fail" and "declined" in case["judge_reasoning"]
    assert not hasattr(judge_mod, "is_abstention") and not hasattr(judge_mod, "judge_case_heuristic")


def test_rag_weak_match_and_duplicate_project(client):
    from finetune_studio.webui.routes.rag import weak_match

    hits = [{"text": "The warranty lasts three years and covers the teapot."}]
    assert weak_match("What is the capital of Mars?", hits)
    assert not weak_match("How long does the warranty last?", hits)
    assert not weak_match("hi", hits)

    assert client.post("/api/projects", json={"name": "Dup Test"}).status_code == 200
    r = client.post("/api/projects", json={"name": "dup test"})
    assert r.status_code == 409 and r.json()["code"] == "duplicate_name"
    assert client.post("/api/projects", json={"name": "dup test", "allow_duplicate": True}).status_code == 200
