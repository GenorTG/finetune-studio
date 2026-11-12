"""Smart Brain retrieval matrix through real exported packages.

    .venv/bin/python tests/corpus/quillmoor/bench/brain_matrix.py --corpus .tmp/quillmoor/s --project agent-test-quillmoor-s \
        --embedders sentence-transformers/all-MiniLM-L6-v2 Qwen/Qwen3-Embedding-0.6B --rerankers none cross-encoder/ms-marco-MiniLM-L-6-v2

For each embedder: the project's index is (re)embedded through the app's build job (POST /brain/build, then
/brain/rebuild-vectors). For each reranker: the Smart Brain page's export (POST /brain/standalone, models bundled,
plain, tar) is unpacked and its own server.py is started three times (meaning search only / word search only /
both + RRF), each with the reranker on when one is bundled. Every answerable question goes to /search; ranks are
scored by retrieval.py. Word search does not depend on the embedder, so it runs once per reranker.
Results: <corpus>/results/brain/<embedder>__<reranker>__<layers>.json and matrix.json.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (
    ROOT,
    check,
    client,
    gpu_serialized,
    load_jsonl,
    post_when_free,
    project_by_name,
    write_json,
)
from llm import GPU_UUID, free_port
from retrieval import evaluate, package_search

PY = ROOT / ".venv" / "bin" / "python"


def slug(s: str) -> str:
    return s.replace("/", "__")


def wait_build(c: httpx.Client, pid: str, r: dict) -> dict:
    bid = r["build_id"]
    while True:
        s = check(c.get(f"/api/projects/{pid}/brain/builds/{bid}"))
        if s.get("finished"):
            if not s.get("ok"):
                raise RuntimeError(f"build {bid} failed: {s.get('error')}")
            return s
        time.sleep(4)


def index_size(pid: str) -> int:
    d = Path.home() / ".finetune-studio" / "brain" / pid
    return sum(p.stat().st_size for p in d.rglob("*") if p.is_file()) if d.is_dir() else 0


def start_server(pkg: Path, flags: list[str], log: Path) -> tuple[subprocess.Popen, str]:
    port = free_port()
    with log.open("w") as logf:
        p = subprocess.Popen([str(PY), "server.py", "--http", str(port), "--port", str(port), *flags], cwd=pkg, stdout=logf,
                             stderr=subprocess.STDOUT, env={**os.environ, "CUDA_VISIBLE_DEVICES": GPU_UUID}, start_new_session=True)
    base = f"http://127.0.0.1:{port}"
    t0 = time.time()
    while True:
        if p.poll() is not None:
            raise RuntimeError(f"package server exited {p.returncode}: {log.read_text()[-800:]}")
        try:
            r = httpx.get(f"{base}/health", timeout=3)
            if r.status_code == 200:
                return p, base
        except httpx.HTTPError:
            pass
        if time.time() - t0 > 600:
            p.kill()
            raise TimeoutError("package server did not start")
        time.sleep(2)


def stop(p: subprocess.Popen) -> None:
    p.terminate()
    try:
        p.wait(30)
    except subprocess.TimeoutExpired:
        p.kill()


def completed_keyword_rerankers(runs: dict[str, dict]) -> set[str]:
    return {str(row["reranker"]) for row in runs.values() if row.get("layers") == "keyword" and row.get("reranker")}


def expected_run_keys(embedders: list[str], rerankers: list[str], layers: list[str]) -> set[str]:
    """Return persisted matrix keys, accounting for keyword search being embedder-independent."""
    expected = ({f"{slug(embedders[0])}__{slug(rr)}__keyword" for rr in rerankers}
                if embedders and "keyword" in layers else set())
    expected.update(
        f"{slug(emb)}__{slug(rr)}__{layer}"
        for emb in embedders
        for rr in rerankers
        for layer in layers
        if layer != "keyword"
    )
    return expected


def complete_run_keys(runs: dict[str, dict], out: Path, expected_rows: int, overall_n: int,
                      expected_ids: set[str]) -> set[str]:
    complete = set()
    for key in runs:
        try:
            result = json.loads((out / f"{key}.json").read_text())
        except (OSError, json.JSONDecodeError):
            continue
        rows = result.get("rows", [])
        ids = [row.get("id") for row in rows]
        if (len(rows) == expected_rows and len(set(ids)) == expected_rows and set(ids) == expected_ids
                and result.get("overall", {}).get("n") == overall_n):
            complete.add(key)
    return complete


def missing_run_keys(complete_keys: set[str], done_keyword: set[str], embedders: list[str], rerankers: list[str],
                     layers: list[str]) -> list[str]:
    missing = []
    if "keyword" in layers:
        missing.extend(f"word__{slug(rr)}__keyword" for rr in rerankers if rr not in done_keyword)
    for emb in embedders:
        for rr in rerankers:
            for layer in layers:
                if layer == "keyword":
                    continue
                key = f"{slug(emb)}__{slug(rr)}__{layer}"
                if key not in complete_keys:
                    missing.append(key)
    return missing


@gpu_serialized
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--project", required=True)
    ap.add_argument("--embedders", nargs="+", required=True)
    ap.add_argument("--rerankers", nargs="+", default=["none"])
    ap.add_argument("--layers", nargs="+", default=["semantic", "keyword", "hybrid"])
    ap.add_argument("--workers", type=int, default=4,
                    help="concurrent search clients (use 1 to isolate package-server stability)")
    ap.add_argument("--phase", choices=["export", "run", "both"], default="both",
                    help="export: re-embed + export every needed package (holds trainings off: GPU lease); "
                         "run: start the exported packages and score them (no lease needed)")
    a = ap.parse_args()
    if a.workers < 1:
        ap.error("--workers must be at least 1")
    corpus = Path(a.corpus)
    qs = load_jsonl(corpus / "eval" / "questions.jsonl")
    out = corpus / "results" / "brain"
    out.mkdir(parents=True, exist_ok=True)
    c = client(timeout=3600)
    proj = project_by_name(c, a.project)
    if not proj:
        raise SystemExit(f"no project {a.project}")
    pid = proj["id"]
    matrix_path = out / "matrix.json"
    matrix = json.loads(matrix_path.read_text()) if matrix_path.exists() else {"builds": {}, "runs": {}}
    matrix["expected_runs"] = sorted(expected_run_keys(a.embedders, a.rerankers, a.layers))
    write_json(matrix_path, matrix)
    answerable_types = {"paraphrase", "verbatim", "superseded", "reverse", "multihop"}
    expected_rows = sum(q["type"] in answerable_types for q in qs)
    overall_n = sum(q["type"] in answerable_types and q["type"] != "verbatim" for q in qs)
    expected_ids = {q["id"] for q in qs if q["type"] in answerable_types}
    complete_keys = complete_run_keys(matrix["runs"], out, expected_rows, overall_n, expected_ids)
    complete_runs = {key: value for key, value in matrix["runs"].items() if key in complete_keys}
    first = not (Path.home() / ".finetune-studio" / "brain" / pid / "manifest.json").exists()
    keyword_done = completed_keyword_rerankers(complete_runs)
    def todo(emb: str, rr: str) -> list[str]:
        return [ly for ly in a.layers if f"{slug(emb)}__{slug(rr)}__{ly}" not in complete_keys
                and not (ly == "keyword" and rr in keyword_done)]

    store = ROOT / ".tmp" / "bench" / ("packages" if corpus.name == "s" else f"packages-{corpus.name}")
    hold = ROOT / ".tmp" / "bench" / "gpu-hold"
    if a.phase in ("export", "both"):
        hold.write_text("brain_matrix export phase")
        try:
            for emb in a.embedders:
                need = [rr for rr in a.rerankers if todo(emb, rr) and not (store / f"{slug(emb)}__{slug(rr)}").is_dir()]
                if not need:
                    continue
                t0 = time.time()
                if first:
                    r = post_when_free(c, f"/api/projects/{pid}/brain/build", json={"embedder": emb, "reset": True})
                    first = False
                else:
                    r = post_when_free(c, f"/api/projects/{pid}/brain/rebuild-vectors", json={"embedder": emb})
                st = wait_build(c, pid, r)
                dur = st.get("duration_ms")    # the build job's own time (t0 includes waiting for the GPU lease)
                matrix["builds"][emb] = {"seconds": round(dur / 1000, 1) if dur else round(time.time() - t0, 1),
                                         "wall_incl_wait_s": round(time.time() - t0, 1), "build_id": r.get("build_id"),
                                         "chunks": st.get("chunks"), "documents": st.get("documents"), "index_bytes": index_size(pid)}
                print(f"[{emb}] embedded in {matrix['builds'][emb]['seconds']}s, {st.get('chunks')} passages", flush=True)
                write_json(matrix_path, matrix)
                for rr in need:
                    if rr != "none":
                        check(c.post(f"/api/projects/{pid}/brain/settings", json={"reranker": rr, "rerank_enabled": True}))
                    t1 = time.time()
                    res = check(c.post(f"/api/projects/{pid}/brain/standalone", json={
                        "include_models": True, "include_reranker": rr != "none", "reranker_enabled": rr != "none",
                        "encrypt": False, "archive_format": "tar", "semantic_search": True, "keyword_search": True,
                        "answer_enabled": False, "name": "qm-bench"}))
                    pkg_file = ROOT / "output" / "projects" / pid / "brain-packages" / res["filename"]
                    dest = store / f"{slug(emb)}__{slug(rr)}"
                    tmp = dest.with_name(dest.name + ".part")
                    shutil.rmtree(tmp, ignore_errors=True)
                    tmp.mkdir(parents=True)
                    with tarfile.open(pkg_file) as tf:
                        tf.extractall(tmp, filter="data")
                    write_json(tmp / "export.json", {"package_bytes": res["size"], "export_s": round(time.time() - t1, 1),
                                                     "embedder": emb, "reranker": rr})
                    tmp.rename(dest)
                    pkg_file.unlink(missing_ok=True)
                    print(f"  exported {dest.name} ({res['size'] / 1e9:.2f} GB)", flush=True)
        finally:
            hold.unlink(missing_ok=True)
    if a.phase in ("run", "both"):
        for emb in a.embedders:
            for rr in a.rerankers:
                dest = store / f"{slug(emb)}__{slug(rr)}"
                if not todo(emb, rr):
                    continue
                if not dest.is_dir():
                    print(f"  no package for {dest.name}: run the export phase first", flush=True)
                    continue
                meta = json.loads((dest / "export.json").read_text())
                pkg = next(p for p in dest.iterdir() if p.is_dir())
                for layers in todo(emb, rr):
                    key = f"{slug(emb)}__{slug(rr)}__{layers}"
                    flags = {"semantic": ["--no-keyword"], "keyword": ["--no-semantic"], "hybrid": []}[layers]
                    flags += ["--no-reranker"] if rr == "none" else ["--reranker"]
                    ev, info, attempts = None, {}, 0
                    while ev is None and attempts < 3:
                        attempts += 1
                        p = None
                        try:
                            p, base = start_server(pkg, flags, ROOT / ".tmp" / "bench" / f"server-{key}-{attempts}.log")
                            info = httpx.get(f"{base}/health", timeout=10).json()
                            t2 = time.time()
                            ev = evaluate(qs, package_search(base), workers=a.workers)
                            ev["seconds"] = round(time.time() - t2, 1)
                        except (httpx.HTTPError, RuntimeError, TimeoutError) as e:
                            # The server died or could not start (often after a CUDA fault); retry it cleanly.
                            print(f"  {key}: server failed ({type(e).__name__}), attempt {attempts}", flush=True)
                        finally:
                            if p is not None:
                                stop(p)
                    if ev is None:
                        print(f"  {key}: no complete score after {attempts} attempts", flush=True)
                        continue
                    ev["attempts"] = attempts
                    ev["workers"] = a.workers
                    rows = ev.pop("rows")
                    write_json(out / f"{key}.json", {"embedder": emb, "reranker": rr, "layers": layers, "mode": info.get("mode"),
                                                      **meta, **ev, "rows": rows})
                    matrix["runs"][key] = {"embedder": emb, "reranker": rr, "layers": layers, "mode": info.get("mode"),
                                           "overall": ev["overall"], "by_type": ev["by_type"], "latency_ms": ev["latency_ms"],
                                           "stale_first": ev["stale_first"], "package_bytes": meta["package_bytes"],
                                           "workers": a.workers}
                    if layers == "keyword":
                        keyword_done.add(rr)
                    complete_keys.add(key)
                    write_json(matrix_path, matrix)
                    o = ev["overall"]
                    print(f"  {layers:8s} {emb.split('/')[-1]:28s} rr={rr.split('/')[-1]:24s} hit@1 {o['hit@1']:.3f} hit@5 {o['hit@5']:.3f} "
                          f"hit@10 {o['hit@10']:.3f} mrr {o['mrr@10']:.3f} p50 {ev['latency_ms'].get('p50')}ms ({info.get('mode')})", flush=True)
    if a.phase in ("run", "both"):
        missing = missing_run_keys(complete_keys, keyword_done, a.embedders, a.rerankers, a.layers)
        if missing:
            print(f"incomplete retrieval matrix: {len(missing)} missing case(s): {missing[:10]}", flush=True)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
