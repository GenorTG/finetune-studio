"""R7: actual encrypted runtime, exact retrieval, durable regression and semantic judge boundaries."""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from finetune_studio.smart_brain import package_qc as qc
from finetune_studio.smart_brain.export_config import BrainExportConfig
from finetune_studio.smart_brain.server_package import build_package
from tests import test_brain_server_package as packages
from tests.test_package_knowledge_api import host, request

CASES = [{"name": "warehouse", "question": "Highmere Stair warehousing cost?",
          "source_id": "c2", "correct_answer": "Nine crowns per season."}]


@pytest.fixture()
def corpus_dir(tmp_path):
    return packages.corpus_dir.__wrapped__(tmp_path)


def fake_corpus():
    corpus = SimpleNamespace(top_k=5, identity={}, cfg={"answer": True, "llm_base_url": "http://operator/v1",
                           "llm_model": "operator-model"},
                           search=lambda question, top_k: [{"document_id": "c2", "text": "Nine crowns per season."}],
                           _chat=lambda messages: "A seasonal charge of 9 crowns.")
    def answer(question, top_k, *, include_sources, trace):
        messages = [{"role": "system", "content": "Use context only: Nine crowns per season."},
                    {"role": "user", "content": question}]
        reply = corpus._chat(messages)
        trace.extend([*messages, {"role": "assistant", "content": reply}])
        return {"answer": reply}
    corpus.answer = answer
    return corpus


def test_exact_scope_validation_and_benchmark_case_contract():
    assert qc.validate_suite({"cases": CASES}, {"c2"})[0]["expected_sources"] == ["c2"]
    for cases in ([{**CASES[0], "source_id": "c"}], [{**CASES[0], "source_id": "highmere.txt"}]):
        with pytest.raises(ValueError, match="outside"):
            qc.validate_suite(cases, {"c2"})
    for value in ({}, "x", [None], CASES * 2, [{**CASES[0], "expected_sources": "c2"}],
                  [{**CASES[0], "expect_abstain": True}], [{**CASES[0], "source_id": ""}],
                  CASES * 101, [{**CASES[0], "question": ""}]):
        with pytest.raises(ValueError):
            qc.validate_suite(value)
    abstain = qc.validate_suite([{"name": "unknown", "question": "Not in corpus?", "expect_abstain": True}])
    assert abstain[0]["expected_sources"] == []


def test_baseline_regression_restart_no_keyword_verdict_and_gate(tmp_path):
    corpus = fake_corpus()
    runner = qc.QualityChecks(tmp_path, CASES)
    first = runner.run(corpus)
    assert first["results"][0]["retrieval_hit"] is True
    assert first["results"][0]["verdict"] == ""
    assert first["gate"]["allowed"] is False
    runner.accept_baseline(first["id"])
    runner = qc.QualityChecks(tmp_path, CASES)
    assert runner.baseline()["id"] == first["id"]
    assert runner.run(corpus)["gate"]["allowed"] is True
    corpus.search = lambda *args: [{"document_id": "c20", "text": "Nine crowns per season."}]
    missed = runner.run(corpus)
    assert missed["results"][0]["retrieval_hit"] is False  # c2 is NOT c20, regardless of answer text
    assert missed["regressions"] == ["warehouse"]
    assert missed["gate"]["allowed"] is False
    with pytest.raises(ValueError, match="without retrieval misses"):
        runner.accept_baseline(missed["id"])
    changed = qc.QualityChecks(tmp_path, [{**CASES[0], "question": "Changed question"}])
    assert "missing_or_incompatible_baseline" in changed.run(corpus)["gate"]["reasons"]
    assert qc.update_gate(None, runner.baseline())["allowed"] is False
    assert (runner.root / "qc.sqlite3").stat().st_mode & 0o777 == 0o600
    assert runner.root.stat().st_mode & 0o777 == 0o700
    runner.clear()
    assert runner.overview()["runs"] == [] and runner.baseline() is None


def test_run_then_judge_paraphrase_partial_invalid_and_recovery(tmp_path):
    runner, corpus = qc.QualityChecks(tmp_path, CASES), fake_corpus()
    run = runner.run(corpus, answers=True)
    row = run["results"][0]
    assert row["model_answer"] == "A seasonal charge of 9 crowns."
    assert row["verdict"] == "" and row["judge"] == "none" and len(row["transcript"]) == 3
    runner.accept_baseline(run["id"])
    assert qc.update_gate(run, runner.baseline(), require_judged=True)["allowed"] is False
    sent = []
    def judge(messages):
        sent.append(messages)
        return '{"verdict":"pass","reasoning":"Faithful paraphrase with the correct amount and period."}'
    corpus._chat = judge
    judged = runner.judge(corpus, run["id"])
    assert judged["results"][0]["verdict"] == "pass"
    assert judged["gate"]["allowed"] is True
    assert "A seasonal charge" in sent[0][1]["content"]
    assert "correct_answer" in sent[0][1]["content"]
    corpus._chat = lambda messages: '{"verdict":"partial","reasoning":"Period missing."}'
    assert runner.judge(corpus, run["id"])["results"][0]["verdict"] == "partial"
    corpus._chat = lambda messages: "Nine crowns per season."  # matching answer key IS NOT a verdict
    failed = runner.judge(corpus, run["id"])
    assert failed["results"][0]["verdict"] == "" and failed["results"][0]["error"] == "judge_failed"
    assert not failed["gate"]["allowed"]
    corpus._chat = judge
    assert runner.judge(corpus, run["id"])["status"] == "complete"


def test_missing_config_provider_failure_abstention_and_bounded_history(tmp_path, monkeypatch):
    runner, corpus = qc.QualityChecks(tmp_path, CASES), fake_corpus()
    corpus.cfg["answer"] = False
    with pytest.raises(qc.MissingModelError):
        runner.run(corpus, answers=True)
    assert runner.overview()["runs"] == []
    assert runner.run(corpus)["status"] == "complete"  # no model is required for retrieval
    corpus.cfg["answer"] = True
    def broken(messages):
        raise RuntimeError("PRIVATE_PROVIDER_KEY")
    corpus._chat = broken
    run = runner.run(corpus, answers=True)
    assert run["status"] == "error" and not run["gate"]["allowed"]
    assert "PRIVATE_PROVIDER_KEY" not in json.dumps(runner.overview())
    assert run["results"][0]["verdict"] == ""
    with pytest.raises(ValueError):
        runner.accept_baseline(run["id"])
    monkeypatch.setattr(qc, "MAX_RUNS", 3)
    for _ in range(5):
        runner.run(corpus)
    assert len(runner.overview()["runs"]) == 3
    assert runner.lock.acquire(False)
    try:
        with pytest.raises(qc.BusyError):
            runner.run(corpus)
        with pytest.raises(qc.BusyError):
            runner.clear()
    finally:
        runner.lock.release()
    abstain = qc.QualityChecks(tmp_path / "abstain", [{"name": "unknown", "question": "Unknown?",
                                                    "expect_abstain": True}])
    abstain_run = abstain.run(corpus)
    assert abstain_run["results"][0]["retrieval_hit"] is None
    assert "no_retrieval_expectations" in abstain_run["gate"]["reasons"]


@pytest.mark.parametrize("encrypted", [False, True])
def test_actual_export_owner_auth_encryption_no_model_and_ui(corpus_dir, tmp_path, encrypted):
    with host(corpus_dir, tmp_path, encrypted=encrypted, qc_cases=CASES) as (base, root):
        assert (root / "qc.html").exists() and (root / "package_qc.py").exists()
        if encrypted:
            assert not (root / "corpus/qc-suite.json").exists()
            for path in root.rglob("*"):
                if path.is_file() and path.name != "corpus.enc":
                    assert b"Highmere Stair warehousing cost?" not in path.read_bytes(), path
        assert request(base, "/qc", token="")[0] == 200
        assert "Run retrieval checks" in request(base, "/qc")[1]
        assert "/qc/run" in request(base, "/qc.js")[1]
        assert request(base, "/qc/report", token="")[0] == 401
        assert request(base, "/qc/report")[0] == 403
        owner = (root / ".usage/owner-token").read_text().strip()
        headers = {"X-Usage-Owner": owner}
        assert request(base, "/qc/report", extra_headers={**headers, "Forwarded": "for=127.0.0.1"})[0] == 403
        assert request(base, "/qc/report", extra_headers={**headers, "Host": "foreign.test"})[0] == 403
        assert request(base, "/qc/run", {}, origin="https://foreign.test", extra_headers=headers)[0] == 403
        report = json.loads(request(base, "/qc/report", extra_headers=headers)[1])
        assert report["cases"][0]["expected_sources"] == ["c2"]
        code, body, _ = request(base, "/qc/run", {}, extra_headers=headers)
        assert code == 200
        run = json.loads(body)
        assert run["results"][0]["retrieval_hit"] is True and not run["gate"]["allowed"]
        assert request(base, "/qc/baseline", {"id": run["id"]}, extra_headers=headers)[0] == 200
        second = json.loads(request(base, "/qc/run", {}, extra_headers=headers)[1])
        assert second["gate"]["allowed"]
        usage_report = json.loads(request(base, "/usage/report", extra_headers=headers)[1])
        assert usage_report["summary"]["questions"] == 0  # synthetic QC must not inflate end-user gaps
        code, body, _ = request(base, "/qc/run", {"answers": True}, extra_headers=headers)
        assert code == 503 and "Configure the operator answer model" in body
        assert request(base, "/qc/run", {"answers": "false"}, extra_headers=headers)[0] == 400
        assert request(base, "/qc/judge", {"id": run["id"], "llm_base_url": "https://evil.test"},
                       extra_headers=headers)[0] == 400
        assert request(base, "/qc/clear", {}, extra_headers=headers)[0] == 200
        assert json.loads(request(base, "/qc/report", extra_headers=headers)[1])["runs"] == []


def test_actual_export_rejects_suite_scope_mismatch(corpus_dir, tmp_path):
    with pytest.raises(ValueError, match="outside"):
        build_package(corpus_dir, tmp_path / "bad.tar.gz", documents={"c1"}, qc_cases=CASES,
                      config=BrainExportConfig(encrypt=True, semantic_search=False, kdf_log_n=10))
    assert not (tmp_path / "bad.tar.gz").exists()
    assert not (tmp_path / "bad.tar.gz.part").exists()


def test_studio_ui_api_and_guide_contract():
    from pathlib import Path

    from finetune_studio.guide import registry
    from finetune_studio.guide.settings_help import SETTING_HELP
    from finetune_studio.webui.routes.brain_export import StandaloneRequest, _patch_values

    req = StandaloneRequest(qc_cases=CASES)
    assert req.qc_cases == CASES and "qc_cases" not in _patch_values(req)
    html = Path("src/finetune_studio/webui/templates/export_models.html").read_text()
    assert 'id="mcp-qc-suite"' in html and "body.qc_cases" in html
    assert "package-qc" in str(registry.CONTROLS)
    assert SETTING_HELP["package_qc_suite"]["kb"] == "package-qc"
