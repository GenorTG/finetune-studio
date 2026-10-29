"""Every Smart Brain export combination, built by the app and run from its own server.py.

    .venv/bin/python tests/corpus/quillmoor/bench/export_combos.py --corpus .tmp/quillmoor/s --project agent-test-quillmoor-s \
        --answer-gguf ~/.finetune-studio/bench-gguf/Qwen__Qwen3.5-4B/model-q8_0.gguf

The live matrix checks all 72 valid combinations (3 search-layer sets × 2 reranker states × 2 answer states × 3 archive
formats × 2 encryption states), rejects each of the 4 no-search combinations, and records 76 checks total. For each
package: its files match its layers (no vectors without meaning search, no BM25 without word search, no plaintext when
encrypted), /health reports the requested layers, 60 questions give the same passages as a plain reference package
with the same layers, and /ask answers when enabled (local llama.cpp answer model). Results:
<corpus>/results/export_combos.json.
"""
from __future__ import annotations

import argparse
import atexit
import itertools
import json
import os
import random
import re
import shutil
import sys
import tarfile
import time
import zipfile
from pathlib import Path
from typing import Any

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent))
from brain_matrix import start_server, stop
from common import (
    ROOT,
    check,
    client,
    gpu_serialized,
    load_jsonl,
    project_by_name,
    write_json,
)
from llm import llama_server

PASS = "quillmoor-benchmark-passphrase-2026"
WORK = ROOT / ".tmp" / "bench" / "combos"
_PACKAGE_ROOT_FILES = frozenset({
    "README.md", "install.sh", "run-http.sh", "run-mcp.sh", "setup.sh", "requirements.txt",
    "rag.config.json", "mcp-config.example.json", "server.py", "brain_container.py",
})
_PUBLIC_MODEL_FILES = frozenset({
    "README.md", "META.json", "config.json", "config_sentence_transformers.json", "model.safetensors",
    "model.safetensors.index.json", "pytorch_model.bin", "pytorch_model.bin.index.json", "model.onnx",
    "model_quantized.onnx", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
    "added_tokens.json", "vocab.json", "vocab.txt", "merges.txt", "spiece.model",
    "sentencepiece.bpe.model", "tokenizer.model", "sentence_bert_config.json", "modules.json",
})
_MODEL_SHARD = re.compile(r"(?:model|pytorch_model)-\d{5}-of-\d{5}\.(?:bin|safetensors)")
_MODULE_CONFIG = re.compile(r"\d+_[A-Za-z0-9_-]+/config\.json")
LAYER_SETS = {"semantic": {"semantic": True, "keyword": False}, "keyword": {"semantic": False, "keyword": True},
              "hybrid": {"semantic": True, "keyword": True}}


def _is_public_model_asset(relative: Path) -> bool:
    """Allow known model weights/configs, never arbitrary files merely because they sit by model weights."""
    parts = relative.parts
    if len(parts) == 1:
        return parts[0] in _PUBLIC_MODEL_FILES or _MODEL_SHARD.fullmatch(parts[0]) is not None
    if len(parts) == 2:
        return _MODULE_CONFIG.fullmatch(relative.as_posix()) is not None or (
            parts[0] == "onnx" and parts[1] in {"model.onnx", "model_quantized.onnx"}
        )
    return False


def combination_jobs() -> list[tuple[str, bool, bool, str, bool]]:
    return [(name, rr, ans, fmt, enc) for name, rr, ans, fmt, enc in itertools.product(
        LAYER_SETS, (False, True), (False, True), ("tar.gz", "tar", "zip"), (True, False))]


def rejection_jobs() -> list[tuple[bool, bool]]:
    """The two search switches off, crossed with reranker and answer switches, must be rejected."""
    return list(itertools.product((False, True), repeat=2))


def combination_key(row: dict[str, Any]) -> tuple[str, bool, bool, str, bool]:
    return (row["layers"], row["reranker"], row["answer"], row["format"], row["encrypted"])


def completed_combinations(results: dict[str, Any]) -> set[tuple[str, bool, bool, str, bool]]:
    return {
        combination_key(row)
        for row in results.get("full", [])
        if row.get("part") == "full" and row.get("ok") is True
    }


def record_combination(results: dict[str, Any], row: dict[str, Any]) -> None:
    rows = results.setdefault(row["part"], [])
    key = combination_key(row)
    rows[:] = [existing for existing in rows if combination_key(existing) != key]
    rows.append(row)


def rejection_key(row: dict[str, Any]) -> tuple[bool, bool]:
    return row["reranker"], row["answer"]


def passed_rejections(results: dict[str, Any]) -> set[tuple[bool, bool]]:
    return {rejection_key(row) for row in results.get("rejections", []) if row.get("ok") is True}


def record_rejection(results: dict[str, Any], row: dict[str, Any]) -> None:
    rows = results.setdefault("rejections", [])
    key = rejection_key(row)
    rows[:] = [existing for existing in rows if rejection_key(existing) != key]
    rows.append(row)


def check_no_search_rejected(c: httpx.Client, pid: str, reranker: bool, answer: bool) -> dict[str, Any]:
    response = c.post(f"/api/projects/{pid}/brain/standalone", json={
        "semantic_search": False, "keyword_search": False,
        "reranker_enabled": reranker, "include_models": reranker, "include_reranker": reranker,
        "answer_enabled": answer, "llm_base_url": "http://127.0.0.1:1234/v1" if answer else "",
        "llm_model": "benchmark-placeholder" if answer else "", "encrypt": False, "archive_format": "tar",
        "name": "qm-invalid-no-search",
    })
    try:
        body = response.json()
    except ValueError:
        body = response.text
    detail = body.get("detail", body) if isinstance(body, dict) else body
    message = str(detail)
    return {"part": "rejections", "reranker": reranker, "answer": answer, "status": response.status_code,
            "detail": message[:500], "ok": response.status_code == 422 and "a package needs one way to find passages" in message}


def export(c, pid: str, **cfg) -> tuple[Path, dict]:
    res = check(c.post(f"/api/projects/{pid}/brain/standalone", json={"name": "qm-combo", **cfg}))
    return ROOT / "output" / "projects" / pid / "brain-packages" / res["filename"], res


def unpack(pkg: Path, dest: Path) -> Path:
    shutil.rmtree(dest, ignore_errors=True)
    dest.mkdir(parents=True)
    if pkg.name.endswith(".zip"):
        with zipfile.ZipFile(pkg) as z:
            z.extractall(dest)
        for p in dest.rglob("*.sh"):
            p.chmod(0o755)
    else:
        with tarfile.open(pkg) as tf:
            tf.extractall(dest, filter="data")
    return next(p for p in dest.iterdir() if p.is_dir())


def files_ok(root: Path, layers: dict, encrypted: bool, reranker: bool) -> list[str]:
    corpus = root / "corpus"
    names = {p.name for p in corpus.iterdir()}
    bad = []
    if encrypted:
        encrypted_container = corpus / "corpus.enc"
        if not encrypted_container.is_file() or encrypted_container.is_symlink():
            bad.append("encrypted package has no regular corpus/corpus.enc container")
        public_model_dirs = set()
        if layers["semantic"]:
            public_model_dirs.add("embedder")
        if reranker:
            public_model_dirs.add("reranker")
        plaintext = []
        for path in corpus.rglob("*"):
            relative = path.relative_to(corpus)
            if path.is_symlink():
                plaintext.append(f"corpus/{relative.as_posix()} (symlink)")
                continue
            if path == encrypted_container or not path.is_file():
                continue
            if (relative.parts[0] in public_model_dirs
                    and _is_public_model_asset(Path(*relative.parts[1:]))):
                continue
            plaintext.append(f"corpus/{relative.as_posix()}")
        for path in root.rglob("*"):
            relative = path.relative_to(root)
            if path.is_symlink():
                plaintext.append(f"{relative.as_posix()} (symlink)")
                continue
            if not path.is_file():
                continue
            if relative.parts[0] == "corpus":
                continue
            if len(relative.parts) == 1 and relative.name in _PACKAGE_ROOT_FILES:
                continue
            plaintext.append(relative.as_posix())
        if plaintext:
            bad.append("encrypted package ships unencrypted payload files: " + ", ".join(sorted(plaintext)[:8]))
    else:
        if layers["semantic"] != ("vectors.npy" in names):
            bad.append(f"vectors.npy present={'vectors.npy' in names} but meaning search={layers['semantic']}")
        if layers["keyword"] != ("bm25.json" in names):
            bad.append(f"bm25.json present={'bm25.json' in names} but word search={layers['keyword']}")
    if reranker != (corpus / "reranker").is_dir():
        bad.append(f"reranker dir present={(corpus / 'reranker').is_dir()} but reranker={reranker}")
    if layers["semantic"] != (corpus / "embedder").is_dir():
        bad.append(f"embedder dir present={(corpus / 'embedder').is_dir()} but meaning search={layers['semantic']}")
    return bad


def health_ok(h: dict, layers: dict, reranker: bool, answer: bool) -> list[str]:
    lay = h.get("layers", {})
    bad = []
    if lay.get("semantic", "").startswith("on") != layers["semantic"]:
        bad.append(f"/health semantic={lay.get('semantic')!r}")
    if (lay.get("keyword") == "on") != layers["keyword"]:
        bad.append(f"/health keyword={lay.get('keyword')!r}")
    if lay.get("reranker", "").startswith("on") != reranker:
        bad.append(f"/health reranker={lay.get('reranker')!r}")
    if lay.get("answer", "").startswith("on") != answer:
        bad.append(f"/health answer={lay.get('answer')!r}")
    return bad


def answer_off_error(status: int, body: dict) -> str | None:
    message = str(body.get("error") or "")
    if status == 404 and "answer generation is off" in message:
        return None
    return f"expected /ask to return 404 explaining answer generation is off; got {status}: {message[:200]}"


def search_all(base: str, qs: list[dict]) -> list[list[str]]:
    out = []
    for q in qs:
        r = httpx.post(f"{base}/search", json={"query": q["question"], "top_k": 5, "dense_extra": 0}, timeout=120)
        r.raise_for_status()
        out.append([h["chunk_id"] for h in r.json()["hits"]])
    return out


@gpu_serialized
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--project", required=True)
    ap.add_argument("--answer-gguf", required=True)
    a = ap.parse_args()
    corpus = Path(a.corpus)
    qs = [q for q in load_jsonl(corpus / "eval" / "questions.jsonl") if q["type"] == "paraphrase"]
    qs = random.Random(3).sample(qs, 60)
    c = client(timeout=3600)
    pid = project_by_name(c, a.project)["id"]
    results_path = corpus / "results" / "export_combos.json"
    results = json.loads(results_path.read_text()) if results_path.exists() else {}
    results.setdefault("layers", [])
    results.setdefault("packaging", [])
    results.setdefault("full", [])
    results.setdefault("rejections", [])
    results.setdefault("references", {})
    refs: dict[str, list[list[str]]] = results["references"]
    jobs = combination_jobs()
    completed = completed_combinations(results)
    pending_jobs = [job for job in jobs if job not in completed]
    rejected = passed_rejections(results)
    for reranker, answer in rejection_jobs():
        if (reranker, answer) in rejected:
            continue
        row = check_no_search_rejected(c, pid, reranker, answer)
        record_rejection(results, row)
        write_json(results_path, results)
        print(f"reject no-search rr={reranker} answer={answer} -> {row['status']} {'OK' if row['ok'] else 'FAIL'} {row['detail'][:140]}",
              flush=True)
        if row["ok"]:
            rejected.add((reranker, answer))
    if not pending_jobs:
        checks_ok = len(completed) + len(passed_rejections(results))
        print(f"{checks_ok}/76 checks passed")
        return 0 if checks_ok == 76 else 1
    keyfile = WORK / "passphrase.txt"
    WORK.mkdir(parents=True, exist_ok=True)
    atexit.register(shutil.rmtree, WORK, ignore_errors=True)
    keyfile.write_text(PASS)
    keyfile.chmod(0o600)
    # Each package gets one synchronous /ask call at a time; a second slot only wastes VRAM alongside its embedder.
    with llama_server(Path(a.answer_gguf).expanduser(), parallel=1) as llm_url:
        for name, rr, ans, fmt, enc in pending_jobs:
            lay = LAYER_SETS[name]
            t0 = time.time()
            cfg = {"semantic_search": lay["semantic"], "keyword_search": lay["keyword"], "reranker_enabled": rr,
                   "include_reranker": rr, "include_models": True, "answer_enabled": ans, "encrypt": enc,
                   "archive_format": fmt, "llm_base_url": llm_url if ans else "", "llm_model": "local" if ans else ""}
            if enc:
                cfg["passphrase"] = PASS
            row = {"part": "full", "layers": name, "reranker": rr, "answer": ans, "format": fmt, "encrypted": enc}
            pkg_file: Path | None = None
            try:
                pkg_file, res = export(c, pid, **cfg)
                row["package_bytes"] = res["size"]
                root = unpack(pkg_file, WORK / "pkg")
                problems = files_ok(root, lay, enc, rr)
                env_before = dict(os.environ)
                p, base = start_server(root, ["--keyfile", str(keyfile)] if enc else [],
                                       ROOT / ".tmp" / "bench" / f"server-combo-full-{name}-{rr}-{ans}-{fmt}-{enc}.log")
                try:
                    h = httpx.get(f"{base}/health", timeout=10).json()
                    row["mode"] = h.get("mode")
                    problems += health_ok(h, lay, rr, ans)
                    hits = search_all(base, qs)
                    key = f"{name}|{rr}"
                    if key in refs:
                        same = sum(x == y for x, y in zip(hits, refs[key]))
                        row["same_hits_as_reference"] = f"{same}/{len(qs)}"
                        if same != len(qs):
                            problems.append(f"search differs from the reference package on {len(qs) - same} questions")
                    else:
                        refs[key] = hits
                        row["same_hits_as_reference"] = "reference"
                    if ans:
                        r = httpx.post(f"{base}/ask", json={"question": qs[0]["question"]}, timeout=300)
                        j = r.json()
                        row["ask_status"] = r.status_code
                        row["ask_answer"] = (j.get("answer") or j.get("error") or "")[:300]
                        if r.status_code != 200 or not j.get("answer"):
                            problems.append(f"/ask failed: {r.status_code} {str(j)[:200]}")
                    else:
                        r = httpx.post(f"{base}/ask", json={"question": qs[0]["question"]}, timeout=60)
                        body = r.json()
                        problem = answer_off_error(r.status_code, body)
                        if problem:
                            problems.append(problem)
                finally:
                    stop(p)
                    os.environ.clear()
                    os.environ.update(env_before)
                row["problems"] = problems
                row["ok"] = not problems
            except Exception as e:  # noqa: BLE001 - recorded per combination
                row["ok"], row["problems"] = False, [f"{type(e).__name__}: {e}"]
            finally:
                if pkg_file is not None:
                    pkg_file.unlink(missing_ok=True)
            row["seconds"] = round(time.time() - t0, 1)
            record_combination(results, row)
            write_json(results_path, results)
            print(f"full      {name:8s} rr={rr!s:5s} answer={ans!s:5s} {fmt:6s} enc={enc!s:5s} -> "
                  f"{'OK' if row['ok'] else 'FAIL'} {row.get('same_hits_as_reference', '')} {row.get('mode', '')} {row['problems'][:2]}",
                  flush=True)
    shutil.rmtree(WORK, ignore_errors=True)
    completed = completed_combinations(results)
    checks_ok = len(completed) + len(passed_rejections(results))
    print(f"{checks_ok}/76 checks passed")
    return 0 if checks_ok == 76 else 1


if __name__ == "__main__":
    raise SystemExit(main())
