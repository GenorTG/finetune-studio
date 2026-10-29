"""Ingest benchmark: upload a Quillmoor tier into a project through the live app and check every value survived parsing.

    .venv/bin/python tests/corpus/quillmoor/bench/ingest.py --corpus .tmp/quillmoor/s --project agent-test-quillmoor-s

Upload goes through POST /api/projects/{pid}/files/upload (the Files page's own call), parsing is the app's own
background parse; the check reads each file's parsed text back from GET .../files/{fid}/parsed and reports both normalized
substring presence and whole-marker matches with hyphens treated as identifier characters (neither is field-aware). Results: <corpus>/results/ingest.json
and per-format tables on stdout. With --skip-upload, upload and parser timings are not remeasured; only parsed text is
fetched again to recompute marker coverage.
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (
    check,
    client,
    contains,
    contains_whole_marker,
    ensure_project,
    load_jsonl,
    norm,
    wait,
    write_json,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--project", required=True, help="project name (created when missing; must start with agent-test-)")
    ap.add_argument("--batch", type=int, default=20)
    ap.add_argument("--skip-upload", action="store_true")
    a = ap.parse_args()
    if not a.project.startswith("agent-test-"):
        raise SystemExit("benchmark projects must be named agent-test-* (cleanup.py removes only those)")
    corpus = Path(a.corpus)
    docs = load_jsonl(corpus / "manifest" / "docs.jsonl")
    c = client(timeout=600)
    proj = ensure_project(c, a.project, f"Quillmoor benchmark corpus ({corpus.name})")
    pid = proj["id"]
    print(f"project {a.project} = {pid}", flush=True)
    t_up0 = time.time()
    if not a.skip_upload:
        existing = {f["original_name"] for f in check(c.get(f"/api/projects/{pid}/files"))["files"]}
        todo = [d for d in docs if Path(d["path"]).name not in existing]
        up_bytes = 0
        for i in range(0, len(todo), a.batch):
            chunk = todo[i:i + a.batch]
            files = []
            for d in chunk:
                p = corpus / "files" / d["path"]
                data = p.read_bytes()
                up_bytes += len(data)
                files.append(("files", (p.name, data)))
            rep = check(c.post(f"/api/projects/{pid}/files/upload", files=files))
            bad = [x for x in rep.get("report", rep.get("files", [])) if x.get("status") == "error"]
            if bad:
                print(f"  upload errors: {bad[:3]}", flush=True)
            if (i // a.batch) % 5 == 0:
                print(f"  uploaded {min(i + a.batch, len(todo))}/{len(todo)} ({up_bytes / 1e6:.1f} MB) {time.time() - t_up0:.0f}s", flush=True)
    t_up = time.time() - t_up0
    files = check(c.get(f"/api/projects/{pid}/files"))["files"]
    by_name = {f["original_name"]: f for f in files}
    t_p0 = time.time()

    def parsed_all():
        st = check(c.get(f"/api/projects/{pid}/files/pipeline"))["status"]
        states = defaultdict(int)
        for v in st.values():
            states[v["status"]] += 1
        busy = states.get("queued", 0) + states.get("parsing", 0)
        print(f"  parse: {dict(states)} ({time.time() - t_p0:.0f}s)", flush=True)
        return st if busy == 0 and len(st) >= len(docs) else None

    st = wait(parsed_all, every=15, timeout=6 * 3600, what="parsing")
    t_parse = time.time() - t_p0
    per_fmt = defaultdict(lambda: {"files": 0, "parsed": 0, "errors": 0, "marks": 0, "found": 0,
                                   "whole_marker": 0, "chars": 0, "missing": []})
    per_file = []
    for d in docs:
        name = Path(d["path"]).name
        row = per_fmt[d["format"]]
        row["files"] += 1
        f = by_name.get(name)
        status = (st.get(f["id"]) or {}).get("status") if f else "missing"
        text = ""
        if status == "ready":
            v = c.get(f"/api/projects/{pid}/files/{f['id']}/parsed")
            if v.status_code == 200:
                text = v.json().get("parsed_md") or ""
        if text:
            row["parsed"] += 1
        else:
            row["errors"] += 1
        row["chars"] += len(text)
        hay = norm(text)
        found = 0
        whole_marker = 0
        for fid, written, _old in d["marks"]:
            ok = contains(hay, written)
            found += ok
            whole_marker += contains_whole_marker(hay, written)
            if not ok and len(row["missing"]) < 8:
                row["missing"].append({"file": name, "fact": fid, "written": written})
        row["marks"] += len(d["marks"])
        row["found"] += found
        row["whole_marker"] += whole_marker
        per_file.append({"file": name, "format": d["format"], "status": status, "chars": len(text), "marks": len(d["marks"]),
                         "found": found, "whole_marker": whole_marker,
                         "parser": (st.get(f["id"]) or {}).get("parser") if f else None,
                         "error": (st.get(f["id"]) or {}).get("error") if f else None})
    tot_m = sum(r["marks"] for r in per_fmt.values())
    tot_f = sum(r["found"] for r in per_fmt.values())
    total_bytes = sum(d["bytes"] or 0 for d in docs)
    out = {"project": a.project, "pid": pid, "corpus": str(corpus), "files": len(docs), "bytes": total_bytes,
           "upload_s": round(t_up, 1), "parse_s": round(t_parse, 1),
           "timing_scope": "marker_reverification_only" if a.skip_upload else "upload_and_parse",
           "verification_s": round(t_parse, 1) if a.skip_upload else None,
           "marks": tot_m, "found": tot_f,
           "survival": round(tot_f / max(tot_m, 1), 4), "per_format": per_fmt, "per_file": per_file}
    write_json(corpus / "results" / "ingest.json", out)
    timing = (f"marker re-verification {t_parse:.0f}s (upload/parser timings not remeasured)" if a.skip_upload
              else f"upload {t_up:.0f}s, parse {t_parse:.0f}s")
    print(f"\n{len(docs)} files, {total_bytes / 1e6:.1f} MB: {timing}; "
          f"marker substrings present {tot_f}/{tot_m} = {100 * tot_f / max(tot_m, 1):.2f}%")
    print(f"{'format':12} {'files':>5} {'parsed':>6} {'markers':>7} {'substring':>10} {'boundary':>9} {'boundary%':>10}")
    for k in sorted(per_fmt):
        r = per_fmt[k]
        print(f"{k:12} {r['files']:5d} {r['parsed']:6d} {r['marks']:7d} {r['found']:10d} {r['whole_marker']:9d} "
              f"{100 * r['whole_marker'] / max(r['marks'], 1):10.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
