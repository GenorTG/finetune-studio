from __future__ import annotations

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import answers as answer_collector
import pytest
import rag_answers
from answers import done_ids
from brain_matrix import (
    complete_run_keys,
    completed_keyword_rerankers,
    expected_run_keys,
    missing_run_keys,
)
from common import contains, contains_whole_marker, gpu_benchmark_lock, norm, write_json
from evaluate import subset
from export_combos import (
    answer_off_error,
    combination_jobs,
    completed_combinations,
    files_ok,
    passed_rejections,
    record_combination,
    record_rejection,
    rejection_jobs,
)
from finetune import (
    ACTIVE_RUN_STATUSES,
    find_adoptable_run,
    live_training_metrics,
    rehydrate_run_result,
    training_metric_fields,
    training_result_ok,
)
from judge import JUDGE_PROTOCOL, case_for, has_valid_judgment, needs_judge, score
from report import answers, archetypes, brain, held_out, public_text, response_latency
from report import export_matrix as export_matrix_report
from report import finetune as finetune_report
from report import ingest as ingest_report
from report import sft_facts as sft_facts_report
from sft_holdout import TRAIN_SAMPLE_SEED, sft_fact_questions
from train_variants import build_rows


def test_completed_run_is_not_adopted_as_a_new_training_attempt() -> None:
    runs = [
        {
            "base_model": "/models/Qwen",
            "data_path": "datasets/quillmoor-train.jsonl",
            "status": "done",
        }
    ]

    assert find_adoptable_run(runs, "/models/Qwen", "quillmoor-train.jsonl") is None


def test_marker_matching_distinguishes_substrings_from_word_boundaries() -> None:
    haystack = norm("device code QM-17; flow rate m³/h = 42")

    assert contains(haystack, "QM-17")
    assert contains_whole_marker(haystack, "QM-17")
    assert contains(haystack, "17")
    assert not contains_whole_marker(haystack, "17")
    assert contains_whole_marker(haystack, "m3/h")


def test_corpus_generator_fails_before_writing_manifests_when_rendering_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import importlib.util

    generator_path = Path(__file__).resolve().parents[1] / "generate.py"
    spec = importlib.util.spec_from_file_location("quillmoor_generate_test", generator_path)
    assert spec is not None and spec.loader is not None
    generator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(generator)
    from qmgen import render

    monkeypatch.setattr(render, "render_all", lambda *_args, **_kwargs: {"unrendered.docx": "renderer failed"})

    result = generator.main(["--tier", "s", "--out", str(tmp_path)])

    assert result == 3
    assert not (tmp_path / "manifest" / "docs.jsonl").exists()
    assert not (tmp_path / "summary.json").exists()


def test_sft_heldout_judge_expects_abstention_without_exposing_the_answer_key() -> None:
    case = case_for({
        "type": "sft-heldout",
        "question": "What is the vendor number of a held-out fictional supplier?",
        "key": "The vendor number is V-00001.",
        "answer": "I don't know.",
        "accept": [],
        "wrong": [],
    })

    assert case.expect_abstain
    assert case.correct_answer == ""


def test_latest_matching_active_run_is_adopted() -> None:
    older = {
        "id": "old",
        "base_model": "/models/Qwen",
        "data_path": "datasets/quillmoor-train.jsonl",
        "status": "running",
    }
    newer = {**older, "id": "new"}

    assert find_adoptable_run([older, newer], "/models/Qwen", "quillmoor-train.jsonl") == newer


def test_training_creation_and_worker_states_remain_active() -> None:
    assert {"created", "loading", "training", "saving"} <= ACTIVE_RUN_STATUSES


def test_live_training_metrics_use_project_scoped_status_not_persisted_run_metrics() -> None:
    snapshot = {
        "status": "training", "project_id": "project-1", "current_step": 2880, "total_steps": 5816,
        "loss": 0.008, "elapsed": 1220.4, "eta": 1244.1, "message": "Training…",
    }

    assert live_training_metrics("training", "project-1", snapshot) == {
        "current_step": 2880,
        "total_steps": 5816,
        "loss": 0.008,
        "elapsed": 1220.4,
        "eta": 1244.1,
        "message": "Training…",
    }
    assert live_training_metrics("done", "project-1", snapshot) == {}
    assert live_training_metrics("training", "project-2", snapshot) == {}


def test_zero_or_non_finite_loss_never_counts_as_a_valid_training_result(tmp_path: Path) -> None:
    gguf = tmp_path / "model.gguf"
    gguf.write_bytes(b"model")

    assert not training_result_ok("done", "", 0.0, [gguf])
    assert not training_result_ok("done", "", float("nan"), [gguf])
    assert not training_result_ok("failed", "CUDA error", 0.3, [gguf])
    assert training_result_ok("done", "", 0.3, [gguf])
    assert not training_result_ok("done", "", 0.3, [gguf], expected_quant="Q4-K-M")

    requested = tmp_path / "model-q4_k_m.gguf"
    requested.write_bytes(b"requested quant")
    assert training_result_ok("done", "", 0.3, [gguf, requested], expected_quant="Q4-K-M")

    empty_gguf = tmp_path / "empty.gguf"
    empty_gguf.touch()
    assert not training_result_ok("done", "", 0.3, [empty_gguf])


def test_training_metric_fields_preserve_real_steps_and_dataset_rows() -> None:
    assert training_metric_fields({"current_step": 972, "total_steps": 972, "elapsed": 611.8}, 4306) == {
        "dataset_rows": 4306,
        "optimizer_steps": 972,
        "planned_steps": 972,
        "training_runtime_seconds": 611.8,
    }
    assert training_metric_fields({"current_step": None, "total_steps": "bad"}, 0) == {
        "dataset_rows": 0,
        "optimizer_steps": None,
        "planned_steps": None,
        "training_runtime_seconds": None,
    }


def test_rehydrate_run_result_recovers_completed_artifact_and_metrics(tmp_path: Path) -> None:
    run_dir = tmp_path / "output" / "run-1"
    gguf = run_dir / "gguf" / "model-q8_0.gguf"
    gguf.parent.mkdir(parents=True)
    gguf.write_bytes(b"merged gguf")
    intermediate = run_dir / "gguf" / "model-f16.gguf"
    intermediate.write_bytes(b"intermediate fp16")
    existing = {"model": "/models/Qwen", "quant": "q8_0", "run_id": "run-1", "settings": {"num_epochs": 4}}
    run = {
        "id": "run-1",
        "status": "done",
        "error": "",
        "output_path": "output/run-1",
        "started_at": 10,
        "finished_at": 20,
        "metrics": {"current_step": 972, "total_steps": 972, "final_loss": 0.3, "eval_loss": 0.4,
                    "duration": 9.5, "elapsed": 8.7},
    }

    result = rehydrate_run_result(existing, run, dataset_rows=4306, root=tmp_path)

    assert result["ok"] is True
    assert result["status"] == "done"
    assert result["run_id"] == "run-1"
    assert result["gguf"] == [str(gguf), str(intermediate)]
    assert result["gguf_size_bytes"] == len(b"merged gguf")
    assert result["optimizer_steps"] == 972
    assert result["planned_steps"] == 972
    assert result["dataset_rows"] == 4306
    assert result["final_loss"] == 0.3
    assert result["eval_loss"] == 0.4
    assert result["training_runtime_seconds"] == 8.7
    assert result["seconds"] == 10
    assert result["settings"] == {"num_epochs": 4}


def test_rehydrate_run_result_does_not_scan_root_without_output_path(tmp_path: Path) -> None:
    unrelated = tmp_path / "unrelated.gguf"
    unrelated.write_bytes(b"not this run's artifact")

    result = rehydrate_run_result(
        {"model": "/models/Qwen", "run_id": "run-2"},
        {"id": "run-2", "status": "done", "metrics": {"final_loss": 0.3}},
        dataset_rows=4306,
        root=tmp_path,
    )

    assert result["gguf"] == []
    assert result["gguf_size_bytes"] is None
    assert result["output_path"] == ""
    assert result["ok"] is False


def test_gpu_benchmark_lock_serializes_threads(tmp_path: Path) -> None:
    lock_path = tmp_path / "gpu-benchmark.lock"
    barrier = threading.Barrier(2)
    state_lock = threading.Lock()
    active = 0
    peak = 0

    def run() -> None:
        nonlocal active, peak
        barrier.wait()
        with gpu_benchmark_lock(lock_path):
            with state_lock:
                active += 1
                peak = max(peak, active)
            time.sleep(0.05)
            with state_lock:
                active -= 1

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(run) for _ in range(2)]
        for future in futures:
            future.result()

    assert peak == 1


def test_failed_answer_rows_are_retried_but_successes_are_not(tmp_path: Path) -> None:
    answers = tmp_path / "answers.jsonl"
    answers.write_text(
        '{"id":"failed","answer":"","error":"timeout"}\n'
        '{"id":"good","answer":"A-17","error":""}\n'
    )

    assert done_ids(answers) == {"good"}


def test_retried_answer_replaces_its_old_error_row_without_duplicate_ids(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    output = tmp_path / "answers.jsonl"
    output.write_text(
        '{"id":"retry","answer":"","error":"timeout"}\n'
        '{"id":"keep","answer":"A-18","error":""}\n'
    )
    question = {
        "id": "retry", "type": "verbatim", "question": "Which code?", "answer": "A-17",
        "accept": ["A-17"], "wrong": [], "formats": [], "etype": "asset", "attr": "serial_no",
    }
    monkeypatch.setattr(answer_collector, "chat", lambda *_args, **_kwargs: "A-17")

    count = answer_collector.collect(
        [question], output, mode="closed", base_url="http://test", workers=1, label="retry"
    )

    rows = [json.loads(line) for line in output.read_text().splitlines() if line.strip()]
    assert count == 1
    assert [row["id"] for row in rows] == ["keep", "retry"]
    assert rows[-1]["answer"] == "A-17" and not rows[-1]["error"]


def test_unreachable_answer_server_aborts_remaining_questions(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    output = tmp_path / "answers.jsonl"
    questions = [
        {
            "id": f"q{i}", "type": "verbatim", "question": f"Question {i}?", "answer": f"A-{i}",
            "accept": [f"A-{i}"], "wrong": [], "formats": [], "etype": "asset", "attr": "serial_no",
        }
        for i in range(3)
    ]
    calls = 0

    def unreachable(*_args: object, **_kwargs: object) -> str:
        nonlocal calls
        calls += 1
        raise answer_collector.httpx.ConnectError("connection refused")

    monkeypatch.setattr(answer_collector, "chat", unreachable)

    with pytest.raises(RuntimeError, match="answer endpoint became unreachable"):
        answer_collector.collect(questions, output, mode="closed", base_url="http://test", workers=1)

    rows = [json.loads(line) for line in output.read_text().splitlines() if line.strip()]
    assert calls == 1
    assert [row["id"] for row in rows] == ["q0"]


def test_score_reports_unique_question_coverage_after_retry() -> None:
    valid = {
        "verdict": "pass",
        "judge_model": "test-judge",
        "benchmark_judge_protocol": JUDGE_PROTOCOL,
        "judge_error": "",
    }
    rows = [
        {"id": "q1", "answer": "", "error": "timeout"},
        {"id": "q1", "answer": "A-17", **valid},
        {"id": "q2", "answer": "A-18"},
        {"id": "q3", "answer": "", "error": "connection reset"},
    ]

    result = score(rows)

    assert result == {
        "n": 3,
        "answered": 2,
        "judged": 1,
        "untrusted_verdicts": 0,
        "answer_coverage": 0.6667,
        "judge_coverage": 0.3333,
        "pass": 1.0,
        "score": 1.0,
    }


def test_response_latency_reports_mean_and_nearest_rank_p90() -> None:
    rows = [{"answer": "ok", "latency": value} for value in (1, 2, 3, 4, 5)]
    rows.extend([
        {"answer": "", "latency": 100},
        {"answer": "ok", "latency": 200, "error": "timeout"},
        {"answer": "ok", "latency": float("nan")},
        {"answer": "ok", "latency": True},
    ])

    assert response_latency(rows) == "3.00 / 5.00"


def test_judge_receives_every_accepted_answer_spelling() -> None:
    case = case_for({
        "type": "paraphrase",
        "question": "Which tag is on the pump?",
        "key": "PMP-4471-B",
        "accept": ["PMP-4471-B", "PMP 4471 B", "pump B"],
        "answer": "The pump is tagged PMP 4471 B.",
    })

    assert case.keywords == ["PMP-4471-B", "PMP 4471 B", "pump B"]


def test_changed_judge_protocol_regrades_successes_but_skips_errors() -> None:
    old = {"answer": "PMP-4471-B", "verdict": "pass", "judge_model": "test-judge",
           "benchmark_judge_protocol": "old"}
    current = {**old, "benchmark_judge_protocol": JUDGE_PROTOCOL}
    failed = {"answer": "", "error": "timeout"}

    assert needs_judge(old)
    assert not needs_judge(current)
    assert needs_judge(current, rejudge=True)
    assert not needs_judge(failed, rejudge=True)


def test_only_current_judge_results_with_valid_identity_and_verdict_are_scored() -> None:
    valid = {
        "answer": "A-17", "verdict": "pass", "judge_model": "test-judge",
        "benchmark_judge_protocol": JUDGE_PROTOCOL, "judge_error": "",
    }
    stale_after_error = {**valid, "judge_error": "provider timeout"}
    rows = [
        {"id": "good", **valid},
        {"id": "label", **valid, "verdict": "excellent"},
        {"id": "stale", **stale_after_error},
        {"id": "identity", **valid, "judge_model": " "},
        {"id": "protocol", **valid, "benchmark_judge_protocol": "old"},
    ]

    assert has_valid_judgment(valid)
    assert all(needs_judge(row) for row in rows[1:])
    assert not needs_judge(valid)
    assert score(rows) == {
        "n": 5,
        "answered": 5,
        "judged": 1,
        "untrusted_verdicts": 4,
        "answer_coverage": 1.0,
        "judge_coverage": 0.2,
        "pass": 1.0,
        "score": 1.0,
    }


def test_answer_disabled_response_must_explain_the_unavailable_layer() -> None:
    assert answer_off_error(404, {"error": "answer generation is off in this server (start it with --answer)"}) is None
    assert answer_off_error(500, {"error": "internal error"}) is not None


def test_export_matrix_covers_full_cross_product_and_retries_failures() -> None:
    jobs = combination_jobs()
    modes = {(name, rr, answer) for name, rr, answer, _fmt, _encrypted in jobs}
    packages = {(fmt, encrypted) for _name, _rr, _answer, fmt, encrypted in jobs}
    assert len(jobs) == len(set(jobs)) == 72
    assert len(modes) == 12
    assert len(packages) == 6

    results = {"layers": [], "packaging": []}
    failed = {"part": "full", "layers": "hybrid", "reranker": False, "answer": False,
              "format": "tar.gz", "encrypted": True, "ok": False}
    record_combination(results, failed)
    assert not completed_combinations(results)

    passed = {**failed, "ok": True}
    record_combination(results, passed)
    key = ("hybrid", False, False, "tar.gz", True)
    assert key in jobs
    assert completed_combinations(results) == {key}
    assert results["full"] == [passed]

    legacy = {**passed, "part": "layers"}
    assert completed_combinations({"layers": [legacy]}) == set()

    assert len(rejection_jobs()) == len(set(rejection_jobs())) == 4
    rejected = {"reranker": False, "answer": True, "status": 422, "ok": True}
    record_rejection(results, rejected)
    assert passed_rejections(results) == {(False, True)}


def test_keyword_matrix_resume_uses_full_reranker_id() -> None:
    runs = {
        "embed__BAAI__bge-reranker-v2-m3__keyword": {
            "reranker": "BAAI/bge-reranker-v2-m3",
            "layers": "keyword",
        }
    }

    assert completed_keyword_rerankers(runs) == {"BAAI/bge-reranker-v2-m3"}
    assert missing_run_keys(set(runs), completed_keyword_rerankers(runs), ["Qwen/Embed"],
                            ["BAAI/bge-reranker-v2-m3"], ["keyword"]) == []


def test_expected_matrix_counts_keyword_once_per_reranker() -> None:
    keys = expected_run_keys(["Qwen/Embed", "BAAI/Embed"], ["none", "rerank/model"], ["keyword", "hybrid"])

    assert len(keys) == 6
    assert "Qwen__Embed__rerank__model__keyword" in keys
    assert "BAAI__Embed__rerank__model__keyword" not in keys
    assert "Qwen__Embed__none__hybrid" in keys


def test_matrix_completion_requires_full_unique_query_rows(tmp_path: Path) -> None:
    runs = {"hybrid": {}}
    write_json(tmp_path / "hybrid.json", {"rows": [{"id": "q1"}, {"id": "q2"}], "overall": {"n": 2}})

    assert complete_run_keys(runs, tmp_path, expected_rows=2, overall_n=2, expected_ids={"q1", "q2"}) == {"hybrid"}

    write_json(tmp_path / "hybrid.json", {"rows": [{"id": "q1"}, {"id": "q1"}], "overall": {"n": 2}})
    assert complete_run_keys(runs, tmp_path, expected_rows=2, overall_n=2, expected_ids={"q1", "q2"}) == set()

    write_json(tmp_path / "hybrid.json", {"rows": [{"id": "q1"}, {"id": "q3"}], "overall": {"n": 2}})
    assert complete_run_keys(runs, tmp_path, expected_rows=2, overall_n=2, expected_ids={"q1", "q2"}) == set()


def test_encrypted_package_rejects_nested_plaintext_payload(tmp_path: Path) -> None:
    root = tmp_path / "package"
    corpus = root / "corpus"
    (corpus / "corpus.enc").parent.mkdir(parents=True)
    (corpus / "corpus.enc").write_bytes(b"encrypted corpus")
    (corpus / "embedder").mkdir()
    (corpus / "embedder" / "model.safetensors").write_bytes(b"public model")
    (corpus / "embedder" / "model-00001-of-00002.safetensors").write_bytes(b"public model shard")
    (corpus / "embedder" / "META.json").write_text("{}")
    (corpus / "embedder" / "README.md").write_text("public model card")
    pooling = corpus / "embedder" / "1_Pooling"
    pooling.mkdir()
    (pooling / "config.json").write_text("{}")
    (corpus / "reranker").mkdir()
    (corpus / "reranker" / "model.safetensors").write_bytes(b"public model")
    (root / "README.md").write_text("package instructions")

    assert files_ok(root, {"semantic": True, "keyword": True}, encrypted=True, reranker=True) == []

    embedded_plaintext = corpus / "embedder" / "bm25.json"
    embedded_plaintext.write_text("plaintext index hidden beside public weights")
    problems = files_ok(root, {"semantic": True, "keyword": True}, encrypted=True, reranker=True)
    assert any("corpus/embedder/bm25.json" in problem for problem in problems)

    embedded_plaintext.unlink()
    nested = corpus / "unexpected" / "nested"
    nested.mkdir(parents=True)
    (nested / "bm25.json").write_text("plaintext index")
    problems = files_ok(root, {"semantic": True, "keyword": True}, encrypted=True, reranker=True)
    assert any("corpus/unexpected/nested/bm25.json" in problem for problem in problems)

    (root / "leaked-chunks.jsonl").write_text("plaintext chunks outside corpus")
    problems = files_ok(root, {"semantic": True, "keyword": True}, encrypted=True, reranker=True)
    assert any("leaked-chunks.jsonl" in problem for problem in problems)

    target = tmp_path / "external-payload.txt"
    target.write_text("payload outside the archive")
    (corpus / "embedder" / "tokenizer.json").symlink_to(target)
    problems = files_ok(root, {"semantic": True, "keyword": True}, encrypted=True, reranker=True)
    assert any("tokenizer.json (symlink)" in problem for problem in problems)


def test_archetype_report_joins_manifest_to_live_file_results(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest" / "docs.jsonl"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(
        '{"path":"ops/a.docx","kind":"contract_doc","marks":[["f1","A-17",false],'
        '["f2","B-22",false]]}\n'
        '{"path":"scans/b.png","kind":"vessel_scan","marks":[["f3","MV-8",false]]}\n'
    )
    results = tmp_path / "results"
    results.mkdir()
    write_json(results / "ingest.json", {"per_file": [
        {"file": "a.docx", "status": "ready", "chars": 120, "found": 1, "whole_marker": 1},
        {"file": "b.png", "status": "ready", "chars": 45, "found": 0, "whole_marker": 0},
    ]})

    report = archetypes([tmp_path])

    assert "| `contract_doc` | 1 | 1/2 (50.0%) | 1/2 (50.0%) |" in report
    assert "| `vessel_scan` | 1 | 0/1 (0.0%) | 0/1 (0.0%) |" in report
    assert "parser output was nonempty for 2/2 files" in report

    write_json(results / "ingest.json", {"per_file": [
        {"file": "a.docx", "status": "ready", "chars": 120, "found": 1},
        {"file": "b.png", "status": "ready", "chars": 45, "found": 0},
    ]})
    legacy = archetypes([tmp_path])

    assert "— (not measured)" in legacy


def test_retrieval_report_shows_quality_and_latency_denominators(tmp_path: Path) -> None:
    run = {
        "layers": "hybrid",
        "embedder": "Qwen/Embed",
        "reranker": "none",
        "overall": {"n": 547, "hit@1": 0.7, "hit@3": 0.8, "hit@5": 0.9, "hit@10": 0.95, "mrr@10": 0.75},
        "by_type": {},
        "stale_first": 0.0,
        "latency_ms": {"p50": 120.0, "p95": 240.0},
        "rows": [{"id": str(i)} for i in range(697)],
    }
    summary = {key: value for key, value in run.items() if key != "rows"}
    eval_dir = tmp_path / "eval"
    eval_dir.mkdir()
    eval_questions = [
        *({"id": f"p{i}", "type": "paraphrase"} for i in range(547)),
        *({"id": f"v{i}", "type": "verbatim"} for i in range(150)),
    ]
    run["rows"] = [{"id": question["id"]} for question in eval_questions]
    (eval_dir / "questions.jsonl").write_text("".join(json.dumps(q) + "\n" for q in eval_questions))
    write_json(tmp_path / "results" / "brain" / "matrix.json", {
        "builds": {}, "runs": {"case": summary}, "expected_runs": ["case", "pending"],
    })
    write_json(tmp_path / "results" / "brain" / "case.json", {
        "rows": run["rows"], "overall": run["overall"], "attempts": 2, "workers": 1,
    })

    result = brain(tmp_path)

    assert "quality n | latency n | tries | clients |" in result
    assert "latency n | tries | clients | hit@1" in result
    assert "| 547 | 697 | 2 | 1 |" in result
    assert "Retrieval matrix: **1/2 scored**; pending: `pending`" in result
    assert "latency `n` includes every answerable search" in result

    write_json(tmp_path / "results" / "brain" / "case.json", {
        "rows": run["rows"], "overall": run["overall"], "attempts": 2,
    })
    inferred = brain(tmp_path)

    assert "| 547 | 697 | 2 | 4* |" in inferred
    assert "legacy per-cell JSON did not save the worker count" in inferred


def test_export_matrix_report_counts_unique_pass_fail_and_pending_checks(tmp_path: Path) -> None:
    path = tmp_path / "results" / "export_combos.json"
    passed = {"part": "full", "layers": "hybrid", "reranker": False, "answer": False, "format": "tar",
              "encrypted": False, "ok": True, "same_hits_as_reference": "60/60",
              "package_bytes": 1_048_576, "seconds": 2.5}
    failed = {**passed, "layers": "semantic", "ok": False}
    rejected = {"reranker": False, "answer": False, "status": 422, "ok": True}
    rejected_failed = {"reranker": True, "answer": True, "status": 500, "ok": False}
    write_json(path, {"full": [passed, failed], "layers": [passed], "rejections": [rejected, rejected_failed]})

    result = export_matrix_report(tmp_path)

    lines = result.splitlines()
    mode_rows = [line for line in lines if line.startswith(("| semantic |", "| keyword |", "| hybrid |"))
                  and line.count("|") == 7]
    assert len(mode_rows) == 12
    assert lines[:2] == [
        "| Search layers | Reranker | Answer | passed | failed | pending |",
        "|---|---|---|---:|---:|---:|",
    ]
    assert "| semantic | off | off | 0/6 | 1 | 5 |" in lines
    assert "| hybrid | off | off | 1/6 | 0 | 5 |" in lines
    assert "| **Valid export total** | — | — | **1/72** | **1** | **70** |" in lines
    assert "| Invalid no-search rejections | 1/4 | 1 | 2 |" in lines
    assert "| **Total checks** | — | — | **2/76** | 2 | 72 |" in lines
    detail_header = "| Search | Reranker | Answer | Archive | Encryption | Status | Search parity | /ask | Package MiB | seconds |"
    detail_rows = [line for line in lines if line.startswith(("| semantic |", "| keyword |", "| hybrid |"))
                   and line.count("|") == 11]
    assert detail_header in lines
    assert len(detail_rows) == 72
    assert "| hybrid | off | off | tar | off | pass | 60/60 | 404 expected | 1.0 | 2.5 |" in detail_rows

    write_json(path, {"layers": [passed], "packaging": [{**passed, "part": "packaging"}]})
    partial = export_matrix_report(tmp_path).splitlines()
    assert "| **Valid export total** | — | — | **0/72** | **0** | **72** |" in partial
    assert "| **Total checks** | — | — | **0/76** | 0 | 76 |" in partial
    assert "| hybrid | off | off | tar | off | partial only | 60/60 | 404 expected | 1.0 | 2.5 |" in partial

    write_json(path, {"full": [{**passed, "part": "full", "ok": False}]})
    failed_full = export_matrix_report(tmp_path).splitlines()
    assert "| **Valid export total** | — | — | **0/72** | **1** | **71** |" in failed_full


def test_answer_report_uses_expected_questions_for_coverage(tmp_path: Path) -> None:
    questions = tmp_path / "eval" / "questions.jsonl"
    questions.parent.mkdir(parents=True)
    all_questions = ([{"id": f"p{i:03}", "type": "paraphrase"} for i in range(141)]
                     + [{"id": f"v{i:03}", "type": "verbatim"} for i in range(61)])
    questions.write_text("".join(json.dumps(q) + "\n" for q in all_questions))
    selected = subset(all_questions, 140, 60)
    first = selected[0]
    answers_path = tmp_path / "results" / "answers" / "partial.jsonl"
    answers_path.parent.mkdir(parents=True)
    answers_path.write_text(json.dumps({
        **first, "answer": "A-1", "verdict": "pass", "latency": 1.0,
        "judge_model": "test-judge", "judge_error": "", "benchmark_judge_protocol": JUDGE_PROTOCOL,
    }) + "\n")

    result = answers(tmp_path)

    assert "| Answer set | expected | records | answered | judged | answer cov | judge cov |" in result
    assert "| partial | 200 | 1 | 1 | 1 | 0.5 | 0.5 | 100.0 |" in result
    assert "100.0 (1/140)" in result
    assert "— (0/60)" in result


def test_held_out_report_uses_expected_fact_partitions(tmp_path: Path) -> None:
    held_path = tmp_path / "results" / "finetune" / "held_out_questions.json"
    held_path.parent.mkdir(parents=True)
    held_path.write_text('["Held question"]')
    train_path = tmp_path / "train" / "train.jsonl"
    train_path.parent.mkdir(parents=True)
    train_path.write_text('{"messages":[{"content":"Held question"}],"fact":"F2"}\n')
    eval_path = tmp_path / "eval" / "questions.jsonl"
    eval_path.parent.mkdir(parents=True)
    eval_path.write_text(
        '{"id":"q1","type":"verbatim","facts":["F1"]}\n'
        '{"id":"q2","type":"paraphrase","facts":["F2"]}\n'
        '{"id":"q3","type":"paraphrase","facts":["F2"]}\n'
    )
    answer_path = tmp_path / "results" / "answers" / "model-tuned__closed.jsonl"
    answer_path.parent.mkdir(parents=True)
    answer_path.write_text(
        '{"id":"q1","type":"verbatim","answer":"A-1","verdict":"pass",'
        '"judge_model":"test-judge","judge_error":"","benchmark_judge_protocol":"' + JUDGE_PROTOCOL + '"}\n'
        '{"id":"q2","type":"paraphrase","answer":"A-2","verdict":"partial",'
        '"judge_model":"test-judge","judge_error":"","benchmark_judge_protocol":"' + JUDGE_PROTOCOL + '"}\n'
    )

    result = held_out(tmp_path)

    assert "| model-tuned | 100.0 (1/1) | 1 | 50.0 (1/2) | 1 |" in result


def test_x3_training_phrasings_stay_in_the_same_fact_split() -> None:
    from finetune_studio.training.data import sft_validation_rows

    base_rows = [
        {"fact": f"fact-{i}", "messages": [{"role": "user", "content": f"Q{i}"},
                                             {"role": "assistant", "content": f"A{i}"}]}
        for i in range(100)
    ]
    facts = {
        row["fact"]: {"id": row["fact"], "attr": "asset:serial_no", "etype": "asset", "entity": row["fact"]}
        for row in base_rows
    }
    x3_rows = build_rows(base_rows, facts)
    validation = sft_validation_rows(x3_rows)
    validation_ids = {id(row) for row in validation}
    training = [row for row in x3_rows if id(row) not in validation_ids]
    validation_facts = {row["fact"] for row in validation}
    training_facts = {row["fact"] for row in training}
    base_validation_facts = {row["fact"] for row in sft_validation_rows(base_rows)}

    assert len(x3_rows) == 3 * len(base_rows)
    assert validation_facts.isdisjoint(training_facts)
    assert validation_facts == base_validation_facts
    assert all(row["pair_id"] == row["fact"] for row in x3_rows)


def test_finetune_report_does_not_present_failed_artifact_as_valid(tmp_path: Path) -> None:
    artifact = tmp_path / "failed.gguf"
    artifact.write_bytes(b"partial artifact")
    results = tmp_path / "results" / "finetune"
    results.mkdir(parents=True)
    write_json(results / "model.json", {
        "model": "family/model",
        "status": "done",
        "ok": False,
        "final_loss": 0.0,
        "eval_loss": None,
        "seconds": 60,
        "gguf": [str(artifact)],
        "output_path": str(tmp_path / "missing-output"),
    })

    result = finetune_report(tmp_path)

    assert "artifact valid" in result
    assert "| `family/model` | `—` | unverified | done | no | — | — | 1.0 | — | 0.0 | None | — | rejected by benchmark validity gate |" in result


def test_finetune_report_reads_optimizer_steps_from_trainer_state(tmp_path: Path) -> None:
    artifact = tmp_path / "model.gguf"
    artifact.write_bytes(b"model")
    output = tmp_path / "run"
    checkpoint = output / "checkpoint-972"
    checkpoint.mkdir(parents=True)
    write_json(checkpoint / "trainer_state.json", {
        "global_step": 972,
        "log_history": [{"train_runtime": 90}],
    })
    results = tmp_path / "results" / "finetune"
    results.mkdir(parents=True)
    write_json(results / "model.json", {
        "model": "family/model",
        "status": "done",
        "ok": True,
        "final_loss": 0.3,
        "eval_loss": 0.4,
        "seconds": 60,
        "dataset_rows": 4306,
        "training_runtime_seconds": 78,
        "train_file": "train.jsonl",
        "gguf": [str(artifact)],
        "output_path": str(output),
    })

    result = finetune_report(tmp_path)

    assert "| `family/model` | `train.jsonl` | unverified | done | yes | 4306 | 972 | 1.0 | 1.3 | 0.3 | 0.4 |" in result

    artifact.unlink()
    write_json(results / "model.json", {
        "model": "family/model",
        "status": "done",
        "ok": True,
        "final_loss": 0.3,
        "eval_loss": 0.4,
        "seconds": 60,
        "dataset_rows": 4306,
        "optimizer_steps": 972,
        "training_runtime_seconds": 78,
        "gguf_size_bytes": 3_420_000_000,
        "train_file": "train.jsonl",
        "gguf": [str(artifact)],
        "output_path": str(output),
    })
    preserved = finetune_report(tmp_path)

    assert "| `family/model` | `train.jsonl` | unverified | done | yes | 4306 | 972 | 1.0 | 1.3 | 0.3 | 0.4 | 3.42 GB |" in preserved


def test_sft_fact_report_requires_exact_question_coverage(tmp_path: Path) -> None:
    train_rows = [
        {"fact": f"fact-{i}", "messages": [{"role": "user", "content": f"Q{i}"},
                                             {"role": "assistant", "content": f"A{i}"}]}
        for i in range(100)
    ]
    train_dir = tmp_path / "train"
    train_dir.mkdir()
    train_path = train_dir / "train.jsonl"
    train_path.write_text("".join(json.dumps(row) + "\n" for row in train_rows))
    questions = sft_fact_questions(train_rows, train_rows)
    results = tmp_path / "results" / "sft-facts"
    results.mkdir(parents=True)
    answer_name = "family-tuned__closed.jsonl"
    scope_counts = {"sft-trained": 0, "sft-heldout": 0}
    answer_rows = []
    for question in questions:
        index = scope_counts[question["type"]]
        scope_counts[question["type"]] += 1
        if question["type"] == "sft-trained":
            verdict = "pass" if index < 5 else "partial"
        else:
            verdict = "partial" if index < 5 else "fail"
        answer_rows.append({"id": question["id"], "fact": question["fact"], "type": question["type"],
                            "question": question["question"], "key": question["answer"],
                            "answer": question["answer"], "verdict": verdict,
                            "judge_model": "deepseek-v4.1-flash", "judge_error": "",
                            "benchmark_judge_protocol": JUDGE_PROTOCOL})
    (results / answer_name).write_text("".join(json.dumps(row) + "\n" for row in answer_rows))
    write_json(results / "family-tuned__summary.json", {
        "schema": "quillmoor-sft-facts-v1",
        "train_file": "train.jsonl",
        "expected_ids": [question["id"] for question in questions],
        "expected_questions": len(questions),
        "expected_trained_facts": scope_counts["sft-trained"],
        "expected_heldout_facts": scope_counts["sft-heldout"],
        "train_sample_seed": TRAIN_SAMPLE_SEED,
        "validation_fact_overlap": 0,
        "ctx_per_slot": 6144,
        "answer_files": [answer_name],
    })

    report = sft_facts_report(tmp_path)

    assert "| `family-tuned` | `train.jsonl` | 6144 | 0/10 (clean) | 20 | — | 50.0%/75.0% (10/10/10) | 0.0%/25.0% (10/10/10) | deepseek-v4.1-flash | 0 | 0 | 0 | yes |" in report
    rogue_row = dict(answer_rows[0], id="not-a-source-question")
    (results / answer_name).write_text("".join(
        json.dumps(row) + "\n" for row in [*answer_rows, rogue_row]
    ))
    rogue_report = sft_facts_report(tmp_path)
    assert "| 20 | — | 50.0%/75.0% (10/10/10) | 0.0%/25.0% (10/10/10) | deepseek-v4.1-flash | 0 | 0 | 0 | no |" in rogue_report

    duplicated_rows = [*answer_rows, answer_rows[0]]
    (results / answer_name).write_text("".join(json.dumps(row) + "\n" for row in duplicated_rows))
    duplicate_report = sft_facts_report(tmp_path)
    assert "| 20 | — | 44.4%/72.2% (9/9/10) | 0.0%/25.0% (10/10/10) | deepseek-v4.1-flash | 0 | 0 | 0 | no |" in duplicate_report

    answer_rows.pop()
    (results / answer_name).write_text("".join(json.dumps(row) + "\n" for row in answer_rows))
    incomplete = sft_facts_report(tmp_path)
    assert "| 20 | — | 50.0%/75.0% (10/10/10) | 0.0%/27.8% (9/9/10) | deepseek-v4.1-flash | 0 | 0 | 0 | no |" in incomplete


def test_sft_fact_report_lists_incomplete_manifest_jobs_without_scores(tmp_path: Path) -> None:
    train_rows = [
        {"fact": f"fact-{i}", "messages": [{"role": "user", "content": f"Q{i}"},
                                               {"role": "assistant", "content": f"A{i}"}]}
        for i in range(100)
    ]
    train_dir = tmp_path / "train"
    train_dir.mkdir()
    (train_dir / "train.jsonl").write_text("".join(json.dumps(row) + "\n" for row in train_rows))
    questions = sft_fact_questions(train_rows, train_rows)
    results = tmp_path / "results" / "sft-facts"
    results.mkdir(parents=True)
    partial_rows = [
        {"id": question["id"], "fact": question["fact"], "question": question["question"],
         "key": question["answer"], "type": question["type"],
         "answer": "A valid partial response" if index == 0 else "",
         "error": "connection refused" if index == 1 else ""}
        for index, question in enumerate(questions[:2])
    ]
    answer_path = results / "sftfacts-qwen35-9b-base__closed.jsonl"
    answer_path.write_text("".join(json.dumps(row) + "\n" for row in partial_rows))
    write_json(results / "eval_manifest.json", {
        "jobs": [{"label": "sftfacts-qwen35-9b-base", "train_file": "train.jsonl",
                  "return_code": -15, "ctx_per_slot": 8192,
                  "log": ".tmp/bench/sftfacts-qwen35-9b-base.log", "scores": {}}]
    })

    report = sft_facts_report(tmp_path)

    assert "**Incomplete evaluations (no score reported):**" in report
    assert "| `sftfacts-qwen35-9b-base` | incomplete (exit -15) | 8192 | 1/20 | 0/20 | 1 | " \
           "`.tmp/bench/sftfacts-qwen35-9b-base.log` |" in report


def test_finetune_report_distinguishes_rowwise_leakage_from_grouped_validation(tmp_path: Path) -> None:
    base_rows = [
        {"fact": f"fact-{i}", "messages": [{"role": "user", "content": f"Q{i}"},
                                             {"role": "assistant", "content": f"A{i}"}]}
        for i in range(100)
    ]
    facts = {
        row["fact"]: {"id": row["fact"], "attr": "asset:serial_no", "etype": "asset", "entity": row["fact"]}
        for row in base_rows
    }
    grouped_rows = build_rows(base_rows, facts)
    rowwise_rows = [{key: value for key, value in row.items() if key != "pair_id"} for row in grouped_rows]
    train_dir = tmp_path / "train"
    train_dir.mkdir()
    (train_dir / "train_x3.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rowwise_rows)
    )
    (train_dir / "train_x3_grouped.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in grouped_rows)
    )
    results = tmp_path / "results" / "finetune"
    results.mkdir(parents=True)
    for name, train_file in (("rowwise", "train_x3.jsonl"), ("grouped", "train_x3_grouped.jsonl")):
        write_json(results / f"{name}.json", {
            "model": f"family/{name}",
            "status": "done",
            "ok": False,
            "train_file": train_file,
            "seconds": 60,
        })

    report = finetune_report(tmp_path)
    rowwise = next(line for line in report.splitlines() if "family/rowwise" in line)
    grouped = next(line for line in report.splitlines() if "family/grouped" in line)

    assert " (leak) |" in rowwise
    assert "| `train_x3_grouped.jsonl` | 0/10 (clean) |" in grouped
    assert "any `leak` is training-pipeline evidence only" in report


def test_ingest_report_exposes_found_count_and_denominator(tmp_path: Path) -> None:
    tier = tmp_path / "s"
    result_path = tier / "results" / "ingest.json"
    write_json(result_path, {"files": 2, "bytes": 100, "upload_s": 1, "parse_s": 2,
                            "marks": 10, "found": 9,
                            "per_format": {".txt": {"files": 2, "marks": 10, "found": 9, "errors": 0}}})

    result = ingest_report([tier])

    assert "| Type | S files | S markers | S substring hits | S boundary hits | S boundary hit % |" in result
    assert "| `.txt` | 2 | 10 | 9 | — | — |" in result


def test_ingest_report_does_not_present_marker_reread_as_parse_timing(tmp_path: Path) -> None:
    tier = tmp_path / "s"
    result_path = tier / "results" / "ingest.json"
    write_json(result_path, {
        "files": 2,
        "bytes": 100,
        "upload_s": 0,
        "parse_s": 7,
        "timing_scope": "marker_reverification_only",
        "marks": 10,
        "found": 9,
        "per_format": {".txt": {"files": 2, "marks": 10, "found": 9, "errors": 0}},
    })

    result = ingest_report([tier])

    assert "marker re-verification only (upload/parser timings not remeasured)" in result
    assert "upload 0 s" not in result


def test_rag_answer_run_stops_retries_after_local_server_dies(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"chat": 0, "health": 0}

    def fail_chat(*args, **kwargs) -> str:
        calls["chat"] += 1
        raise rag_answers.httpx.ConnectError("connection refused")

    def fail_health(*args, **kwargs):
        calls["health"] += 1
        raise rag_answers.httpx.ConnectError("connection refused")

    monkeypatch.setattr(rag_answers, "chat", fail_chat)
    monkeypatch.setattr(rag_answers.httpx, "get", fail_health)
    monkeypatch.setattr(rag_answers.time, "sleep", lambda _seconds: None)

    questions = [
        {"id": f"q{i}", "type": "fact", "question": f"Question {i}?", "answer": f"A-{i}",
         "accept": [], "wrong": [], "formats": [], "etype": "txt", "attr": "value"}
        for i in (1, 2)
    ]
    output = tmp_path / "rag.jsonl"

    rag_answers.answer_all(
        questions,
        None,
        output,
        base_url="http://127.0.0.1:12345/v1",
        model="local",
        api_key="",
        label="test",
        workers=1,
        server_health_url="http://127.0.0.1:12345/health",
    )

    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert calls == {"chat": 1, "health": 1}
    assert len(rows) == 2
    assert "connection refused" in rows[0]["error"]
    assert "server unavailable" in rows[1]["error"]


def test_public_text_hides_checkout_and_home_paths() -> None:
    root, home = Path("/srv/u/work/fts"), Path("/srv/u")
    text = "| `/srv/u/work/fts/models/x` | (/srv/u/work/fts/.venv/lib) | /srv/u/.cache/hf | /srv/u |"

    assert public_text(text, repo_root=root, home=home) == "| `models/x` | (.venv/lib) | ~/.cache/hf | ~ |"
