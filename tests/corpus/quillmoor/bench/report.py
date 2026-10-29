"""Render every Quillmoor result file into Markdown tables (no number is copied by hand).

    .venv/bin/python tests/corpus/quillmoor/bench/report.py --tiers .tmp/quillmoor/s .tmp/quillmoor/m > .tmp/quillmoor/tables.md
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from brain_matrix import complete_run_keys
from common import load_jsonl
from evaluate import subset
from export_combos import (
    combination_jobs,
    combination_key,
    completed_combinations,
    passed_rejections,
    rejection_jobs,
    rejection_key,
)
from judge import has_valid_judgment, score
from sft_holdout import TRAIN_SAMPLE_SEED, sft_fact_questions

TYPES = ("paraphrase", "verbatim", "superseded", "multihop", "reverse", "unanswerable")


def response_latency(rows: list[dict[str, Any]]) -> str:
    """Summarize successful model response latency as mean / p90 seconds."""
    values = sorted(
        float(row["latency"])
        for row in rows
        if not row.get("error")
        and str(row.get("answer") or "").strip()
        and isinstance(row.get("latency"), (int, float))
        and not isinstance(row.get("latency"), bool)
        and math.isfinite(float(row["latency"]))
        and row["latency"] >= 0
    )
    if not values:
        return "—"
    p90 = values[math.ceil(0.9 * len(values)) - 1]
    return f"{statistics.fmean(values):.2f} / {p90:.2f}"


def pct(x: float | None) -> str:
    return "—" if x is None else f"{100 * x:.1f}"


def ingest(tiers: list[Path]) -> str:
    rows = {}
    for t in tiers:
        p = t / "results" / "ingest.json"
        if p.exists():
            rows[t.name] = json.loads(p.read_text())
    if not rows:
        return ""
    names = list(rows)
    fmts = sorted({f for r in rows.values() for f in r["per_format"]})
    out = ["| Type | " + " | ".join(f"{n.upper()} files | {n.upper()} markers | {n.upper()} substring hits | "
                                           f"{n.upper()} boundary hits | {n.upper()} boundary hit %"
                                      for n in names) + " |",
           "|---|" + "---|---|---|---|---|" * len(names)]
    for f in fmts:
        cells = []
        for n in names:
            v = rows[n]["per_format"].get(f)
            if v:
                cells += [str(v["files"]), str(v["marks"]), str(v["found"]),
                          str(v.get("whole_marker", "—")),
                          pct(v["whole_marker"] / max(v["marks"], 1)) if "whole_marker" in v else "—"]
            else:
                cells += ["", "", "", "", ""]
        out.append(f"| `{f}` | " + " | ".join(cells) + " |")
    tot = []
    for n in names:
        r = rows[n]
        tot += [str(r["files"]), str(r["marks"]), str(r["found"]),
                str(r.get("whole_marker", "—")),
                pct(r["whole_marker"] / max(r["marks"], 1)) if "whole_marker" in r else "—"]
    out.append("| **all** | " + " | ".join(f"**{c}**" for c in tot) + " |")
    out.append("")
    for n in names:
        r = rows[n]
        failures = sum(v["errors"] for v in r["per_format"].values())
        if r.get("timing_scope") == "marker_reverification_only" or r.get("upload_s") == 0:
            timing = "marker re-verification only (upload/parser timings not remeasured)"
        else:
            timing = f"upload {r['upload_s']:.0f} s, parse finished {r['parse_s']:.0f} s after upload"
        out.append(f"- {n.upper()}: {r['files']} files, {r['bytes'] / 1e6:.1f} MB; {timing}; "
                   f"files that failed to parse: {failures}")
    out.append("Boundary hits treat hyphens as part of identifiers; they are not field-aware, and a dash means the older result did not record them.")
    return "\n".join(out)


def archetypes(tiers: list[Path]) -> str:
    """Join generated document kinds to the live ingest's per-file marker-survival results."""
    by_tier: dict[str, dict[str, dict[str, int | bool]]] = {}
    parsed_by_tier: dict[str, tuple[int, int]] = {}
    for tier in tiers:
        result_path = tier / "results" / "ingest.json"
        manifest_path = tier / "manifest" / "docs.jsonl"
        if not result_path.exists() or not manifest_path.exists():
            continue
        result = json.loads(result_path.read_text())
        files = {str(row["file"]): row for row in result.get("per_file", [])}
        kinds: dict[str, dict[str, int | bool]] = defaultdict(
            lambda: {"files": 0, "parsed": 0, "marks": 0, "found": 0, "whole_marker": 0,
                     "boundary_measured": True}
        )
        for doc in load_jsonl(manifest_path):
            kind = str(doc["kind"])
            row = files.get(Path(doc["path"]).name)
            stat = kinds[kind]
            stat["files"] += 1
            stat["marks"] += len(doc.get("marks", []))
            stat["found"] += int(row.get("found", 0)) if row else 0
            if row and "whole_marker" in row:
                stat["whole_marker"] += int(row["whole_marker"])
            else:
                stat["boundary_measured"] = False
            if row and row.get("status") == "ready" and int(row.get("chars", 0)) > 0:
                stat["parsed"] += 1
        by_tier[tier.name] = kinds
        parsed_by_tier[tier.name] = (
            sum(stat["parsed"] for stat in kinds.values()),
            sum(stat["files"] for stat in kinds.values()),
        )
    if not by_tier:
        return ""

    names = list(by_tier)
    kinds = sorted({kind for values in by_tier.values() for kind in values})
    out = ["| Archetype | " + " | ".join(
        f"{name.upper()} files | {name.upper()} substring hits | {name.upper()} boundary hits" for name in names
    ) + " |", "|---|" + "---|---|---|" * len(names)]

    def recovery(found: int | None, marks: int) -> str:
        if found is None:
            return "— (not measured)"
        return f"{found}/{marks} ({pct(found / marks)}%)" if marks else "— (0 markers)"

    for kind in kinds:
        cells = []
        for name in names:
            stat = by_tier[name].get(kind)
            if stat:
                boundary_hits = stat["whole_marker"] if stat["boundary_measured"] else None
                cells.extend([str(stat["files"]), recovery(stat["found"], stat["marks"]),
                              recovery(boundary_hits, stat["marks"])])
            else:
                cells.extend(["", "", ""])
        out.append(f"| `{kind}` | " + " | ".join(cells) + " |")
    totals = []
    for name in names:
        stats = by_tier[name].values()
        files = sum(stat["files"] for stat in stats)
        marks = sum(stat["marks"] for stat in by_tier[name].values())
        found = sum(stat["found"] for stat in by_tier[name].values())
        stats = list(by_tier[name].values())
        whole_marker = (sum(stat["whole_marker"] for stat in stats)
                        if all(stat["boundary_measured"] for stat in stats) else None)
        totals.extend([str(files), recovery(found, marks), recovery(whole_marker, marks)])
    out.append("| **all** | " + " | ".join(f"**{cell}**" for cell in totals) + " |")
    out.append("")
    out.extend(f"- {name.upper()} parser output was nonempty for {parsed}/{total} files."
               for name, (parsed, total) in parsed_by_tier.items())
    out.append("Boundary hits treat hyphens as part of identifiers; they are not field-aware, and a dash means not measured in this artifact.")
    return "\n".join(out)


def retrieval_type_cell(run: dict[str, Any], name: str) -> str:
    values = run["by_type"].get(name)
    return f"{pct(values['hit@5'])} ({values['n']})" if values else "—"


def brain(tier: Path) -> str:
    p = tier / "results" / "brain" / "matrix.json"
    if not p.exists():
        return ""
    m = json.loads(p.read_text())
    out = ["| Embedder | build s | passages |", "|---|---|---|"]
    for emb, b in m["builds"].items():
        out.append(f"| `{emb}` | {b['seconds']} | {b['chunks']} |")
    expected = set(m.get("expected_runs", []))
    if expected:
        questions = load_jsonl(tier / "eval" / "questions.jsonl")
        answerable_types = {"paraphrase", "verbatim", "superseded", "reverse", "multihop"}
        expected_rows = sum(q["type"] in answerable_types for q in questions)
        overall_n = sum(q["type"] in answerable_types and q["type"] != "verbatim" for q in questions)
        expected_ids = {q["id"] for q in questions if q["type"] in answerable_types}
        complete = expected & complete_run_keys(m["runs"], p.parent, expected_rows, overall_n, expected_ids)
        pending = sorted(expected - complete)
        status = f"Retrieval matrix: **{len(complete)}/{len(expected)} scored**"
        if pending:
            status += "; pending: " + ", ".join(f"`{key}`" for key in pending)
        out += ["", status]
    else:
        out += ["", f"Retrieval matrix completeness not recorded ({len(m['runs'])} result rows)."]
    header = (
        "| Embedder | Reranker | Layers | quality n | latency n | tries | clients | hit@1 | hit@3 | hit@5 | hit@10 | MRR@10 | "
        "para@5 (n) | verbatim@5 (n) | changed@5 (n) | reverse@5 (n) | multi-hop@5 (n) | stale-first (n) | p50 ms | p95 ms |"
    )
    out += ["", header, "|" + "|".join("---" for _ in header[1:-1].split("|")) + "|"]
    inferred_workers = False
    for _k, r in sorted(m["runs"].items(), key=lambda kv: -kv[1]["overall"]["hit@5"]):
        o = r["overall"]
        emb = "—" if r["layers"] == "keyword" else f"`{r['embedder'].split('/')[-1]}`"
        stale_n = r["by_type"].get("superseded", {}).get("n", 0)
        rows = r.get("rows")
        artifact: dict[str, Any] = {}
        try:
            artifact = json.loads((p.parent / f"{_k}.json").read_text())
        except (OSError, json.JSONDecodeError):
            pass
        if not isinstance(rows, list) or not rows:
            rows = artifact.get("rows", [])
        latency_n = len(rows) if isinstance(rows, list) else 0
        attempts = artifact.get("attempts", "—")
        workers = artifact.get("workers")
        if workers is None:
            workers = "4*"
            inferred_workers = True
        out.append(f"| {emb} | {r['reranker'].split('/')[-1]} | {r['layers']} | {o['n']} | {latency_n} | "
                   f"{attempts} | {workers} | {pct(o['hit@1'])} | {pct(o['hit@3'])} | {pct(o['hit@5'])} | {pct(o['hit@10'])} | "
                   f"{pct(o['mrr@10'])} | {retrieval_type_cell(r, 'paraphrase')} | "
                   f"{retrieval_type_cell(r, 'verbatim')} | {retrieval_type_cell(r, 'superseded')} | "
                   f"{retrieval_type_cell(r, 'reverse')} | {retrieval_type_cell(r, 'multihop')} | "
                   f"{pct(r['stale_first'])} ({stale_n}) | {r['latency_ms'].get('p50')} | {r['latency_ms'].get('p95')} |")
    if inferred_workers:
        out += ["", "`4*` is inferred from the recorded driver invocation and its default; legacy per-cell JSON did not save the worker count."]
    out += ["", "Quality `n` excludes verbatim probes; latency `n` includes every answerable search, including verbatim queries."]
    return "\n".join(out)


def export_matrix(tier: Path) -> str:
    path = tier / "results" / "export_combos.json"
    if not path.exists():
        return ""
    results = json.loads(path.read_text())
    expected = set(combination_jobs())
    passed = completed_combinations(results) & expected
    attempted = {
        combination_key(row)
        for row in results.get("full", [])
        if row.get("part") == "full" and combination_key(row) in expected
    }
    failed = attempted - passed
    pending = expected - passed - failed

    rejected_expected = set(rejection_jobs())
    rejected_passed = passed_rejections(results) & rejected_expected
    rejected_attempted = {
        rejection_key(row) for row in results.get("rejections", []) if rejection_key(row) in rejected_expected
    }
    rejected_failed = rejected_attempted - rejected_passed
    rejected_pending = rejected_expected - rejected_passed - rejected_failed
    out = [
        "| Search layers | Reranker | Answer | passed | failed | pending |",
        "|---|---|---|---:|---:|---:|",
    ]
    for layers in ("semantic", "keyword", "hybrid"):
        for reranker in (False, True):
            for answer in (False, True):
                group = {job for job in expected if job[:3] == (layers, reranker, answer)}
                group_passed = passed & group
                group_failed = failed & group
                group_pending = pending & group
                out.append(f"| {layers} | {'on' if reranker else 'off'} | {'on' if answer else 'off'} | "
                           f"{len(group_passed)}/{len(group)} | {len(group_failed)} | {len(group_pending)} |")
    out.append(f"| **Valid export total** | — | — | **{len(passed)}/{len(expected)}** | "
               f"**{len(failed)}** | **{len(pending)}** |")
    rejected_row = (f"| Invalid no-search rejections | {len(rejected_passed)}/{len(rejected_expected)} | "
                    f"{len(rejected_failed)} | {len(rejected_pending)} |")
    total_row = (f"| **Total checks** | — | — | **{len(passed) + len(rejected_passed)}/"
                 f"{len(expected) + len(rejected_expected)}** | "
                 f"{len(failed) + len(rejected_failed)} | {len(pending) + len(rejected_pending)} |")
    out.extend(["", "Each valid mode row contains six packages: three archive formats × encryption on/off.",
                "", "| Additional check set | passed | failed | pending |", "|---|---:|---:|---:|"])
    out.extend([rejected_row, total_row])

    result_rows: dict[tuple[str, bool, bool, str, bool], dict[str, Any]] = {}
    for part in ("layers", "packaging", "full"):
        for row in results.get(part, []):
            key = combination_key(row)
            if key in expected:
                result_rows[key] = row
    out.extend([
        "",
        "| Search | Reranker | Answer | Archive | Encryption | Status | Search parity | /ask | Package MiB | seconds |",
        "|---|---|---|---|---|---|---|---|---:|---:|",
    ])
    for key in combination_jobs():
        layers, reranker, answer, archive_format, encrypted = key
        row = result_rows.get(key)
        if key in passed:
            status = "pass"
        elif key in attempted:
            status = "fail"
        elif row is not None:
            status = "partial only"
        else:
            status = "pending"
        parity_value = str(row.get("same_hits_as_reference") or "—") if row else "—"
        parity = "baseline (60)" if parity_value == "reference" else parity_value
        if row is None:
            ask = "—"
        elif answer and row.get("ask_status") is not None:
            ask = f"{row['ask_status']} answer"
        elif answer:
            ask = "answer on" if row.get("ok") is True else "see failure"
        else:
            ask = "404 expected" if row.get("ok") is True else "see failure"
        package_bytes = row.get("package_bytes") if row else None
        package_mib = f"{package_bytes / (1024 * 1024):.1f}" if isinstance(package_bytes, (int, float)) else "—"
        seconds = row.get("seconds") if row else None
        seconds_text = f"{seconds:.1f}" if isinstance(seconds, (int, float)) else "—"
        out.append(f"| {layers} | {'on' if reranker else 'off'} | {'on' if answer else 'off'} | {archive_format} | "
                   f"{'on' if encrypted else 'off'} | {status} | {parity} | {ask} | {package_mib} | {seconds_text} |")
    return "\n".join(out)


def score_cell(rows: list[dict], expected_n: int) -> str:
    result = score(rows)
    return f"{pct(result.get('score'))} ({result['judged']}/{expected_n})"


def answer_questions(tier: Path) -> list[dict[str, Any]]:
    """Return the exact deterministic subset used by evaluate.py and rag_answers.py."""
    path = tier / "eval" / "questions.jsonl"
    return subset(load_jsonl(path), 140, 60) if path.exists() else []


def answers(tier: Path) -> str:
    d = tier / "results" / "answers"
    eval_path = tier / "eval" / "questions.jsonl"
    if not d.is_dir() or not eval_path.exists():
        return ""
    expected = {str(q["id"]): q for q in answer_questions(tier)}
    if not expected:
        return ""
    expected_by_type = {name: sum(q["type"] == name for q in expected.values()) for name in TYPES}
    out = ["| Answer set | expected | records | answered | judged | answer cov | judge cov | score¹ | p50 s | p95 s | "
           + " | ".join(TYPES) + " |",
           "|" + "|".join("---" for _ in range(10 + len(TYPES))) + "|"]
    for f in sorted(d.glob("*.jsonl")):
        rows = [r for r in load_jsonl(f) if str(r.get("id", "")) in expected]
        s = score(rows)
        cells = [score_cell([r for r in rows if r["type"] == t], expected_by_type[t]) for t in TYPES]
        records = len({str(r["id"]) for r in rows})
        current = {str(r["id"]): r for r in rows if not r.get("error") and str(r.get("answer") or "").strip()}
        latencies = sorted(float(r["latency"]) for r in current.values() if r.get("latency") is not None)
        p50 = round(statistics.median(latencies), 2) if latencies else "—"
        p95 = round(latencies[int(0.95 * (len(latencies) - 1))], 2) if latencies else "—"
        answer_coverage = s["answered"] / len(expected)
        judge_coverage = s["judged"] / len(expected)
        out.append(f"| {f.stem} | {len(expected)} | {records} | {s['answered']} | {s['judged']} | "
                   f"{pct(answer_coverage)} | {pct(judge_coverage)} | {pct(s.get('score'))} | {p50} | {p95} | "
                   + " | ".join(cells) + " |")
    out.extend(["", ("¹ Pass = 1, partial = 0.5, fail = 0; score is over judged answers. Coverage uses the fixed 376-question "
                     "answer subset (140 paraphrase, 60 verbatim, and every multi-hop, reverse, superseded, and unanswerable "
                     "question); `records` counts unique expected question IDs present in the result file.")])
    return "\n".join(out)


def validation_fact_leakage(tier: Path, train_file: str | None) -> str:
    """Report validation facts also present in training, using the app's exact split implementation."""
    if not isinstance(train_file, str) or not train_file.strip():
        return "unverified"
    dataset_path = tier / "train" / Path(train_file).name
    if not dataset_path.is_file():
        return "unverified"
    rows = [json.loads(line) for line in dataset_path.read_text().splitlines() if line.strip()]
    if not rows or any(not isinstance(row, dict) or not isinstance(row.get("fact"), str) for row in rows):
        return "unverified"

    from finetune_studio.training.data import sft_validation_rows

    validation = sft_validation_rows(rows)
    if not validation:
        return "unverified"
    validation_ids = {id(row) for row in validation}
    training = [row for row in rows if id(row) not in validation_ids]
    validation_facts = {row["fact"] for row in validation}
    training_facts = {row["fact"] for row in training}
    if not validation_facts:
        return "unverified"
    leaked = len(validation_facts & training_facts)
    label = "leak" if leaked else "clean"
    return f"{leaked}/{len(validation_facts)} ({label})"


def finetune(tier: Path) -> str:
    d = tier / "results" / "finetune"
    if not d.is_dir():
        return ""
    out = ["| Base model | training file | validation fact overlap | status | artifact valid | dataset rows | steps | job min | training min | final loss | eval loss | validated GGUF | detail |",
           "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for f in sorted(d.glob("*.json")):
        if f.name.startswith(("dataset", "held_out_questions")):
            continue
        r = json.loads(f.read_text())
        g = r.get("gguf") or []
        gguf_size_bytes = r.get("gguf_size_bytes")
        if gguf_size_bytes is None and r.get("ok") and g and Path(g[0]).is_file():
            gguf_size_bytes = Path(g[0]).stat().st_size
        size = f"{gguf_size_bytes / 1e9:.2f} GB" if r.get("ok") and gguf_size_bytes else "—"
        job_minutes = round((r.get("seconds") or 0) / 60, 1)
        training_minutes: float | str = "—"
        training_runtime = r.get("training_runtime_seconds")
        if isinstance(training_runtime, (int, float)) and math.isfinite(training_runtime) and training_runtime >= 0:
            training_minutes = round(training_runtime / 60, 1)
        loss, ev = r.get("final_loss"), r.get("eval_loss")
        steps = r.get("optimizer_steps")
        dataset_rows = r.get("dataset_rows")
        states = sorted(Path(r.get("output_path") or "/nonexistent").glob("checkpoint-*/trainer_state.json"))
        if states:   # the trainer's own log: runtime, last train loss, last eval loss
            trainer_state = json.loads(states[-1].read_text())
            steps = trainer_state.get("global_step", steps)
            hist = trainer_state.get("log_history", [])
            losses = [h["loss"] for h in hist if "loss" in h]
            evals = [h["eval_loss"] for h in hist if "eval_loss" in h]
            runtime = next((h["train_runtime"] for h in reversed(hist) if "train_runtime" in h), None)
            loss = losses[-1] if losses else loss
            ev = evals[-1] if evals else ev
            if training_minutes == "—" and runtime:
                training_minutes = round(runtime / 60, 1)
        detail = str(r.get("error") or "").replace("\n", " ").replace("|", "/").strip()
        if not r.get("ok") and not detail:
            detail = "rejected by benchmark validity gate"
        detail = detail[:120] or "—"
        step_text = "—" if steps is None else str(steps)
        rows_text = "—" if dataset_rows is None else str(dataset_rows)
        train_file = str(r.get("train_file") or "—").replace("|", "/")
        overlap = validation_fact_leakage(tier, r.get("train_file"))
        out.append(f"| `{r['model']}` | `{train_file}` | {overlap} | {r.get('status')} | {'yes' if r.get('ok') else 'no'} | {rows_text} | {step_text} | "
                   f"{job_minutes} | {training_minutes} | {loss if loss is None else round(loss, 3)} | "
                   f"{ev if ev is None else round(ev, 3)} | {size} | {detail} |")
    out.extend(["", "Validation fact overlap uses the app's exact SFT split; values are overlapping held-out facts / held-out facts. `clean` supports generalization scoring; any `leak` is training-pipeline evidence only; missing provenance is `unverified`.",
                "Job time is API wall duration; training time is the API's elapsed training phase (falling back to Transformers' train_runtime). Job time includes model loading, merge, and GGUF export."])
    return "\n".join(out)


def held_out(tier: Path) -> str:
    """Closed-book verbatim/paraphrase score of tuned models on facts the trainer saw vs. the 10 % it held out."""
    hq = tier / "results" / "finetune" / "held_out_questions.json"
    d = tier / "results" / "answers"
    if not hq.exists() or not d.is_dir():
        return ""
    held = set(json.loads(hq.read_text()))
    facts_held = set()
    qs = {q["id"]: q for q in answer_questions(tier)}
    train = load_jsonl(tier / "train" / "train.jsonl")
    for row in train:
        if row["messages"][0]["content"] in held:
            facts_held.add(row["fact"])
    expected = [q for q in qs.values() if q["type"] in ("verbatim", "paraphrase")]
    expected_seen = [q for q in expected if not set(q["facts"]) & facts_held]
    expected_unseen = [q for q in expected if set(q["facts"]) & facts_held]
    out = ["| Tuned model (closed book) | trained facts: score (judged/total) | n | held-out facts: score (judged/total) | n |",
           "|---|---|---|---|---|"]
    for f in sorted(d.glob("*-tuned__closed.jsonl")):
        rows = [r for r in load_jsonl(f) if r["id"] in qs and r["type"] in ("verbatim", "paraphrase")]
        seen = [r for r in rows if not set(qs[r["id"]]["facts"]) & facts_held]
        unseen = [r for r in rows if set(qs[r["id"]]["facts"]) & facts_held]
        out.append(f"| {f.stem.replace('__closed', '')} | {score_cell(seen, len(expected_seen))} | {score(seen)['n']} | "
                   f"{score_cell(unseen, len(expected_unseen))} | {score(unseen)['n']} |")
    return "\n".join(out)


def sft_facts(tier: Path) -> str:
    """Report matched seen/held-out SFT fact scores with judge and exact-ID coverage."""
    results = tier / "results" / "sft-facts"
    if not results.is_dir():
        return ""
    summaries = sorted(results.glob("*__summary.json"))
    manifest_path = results / "eval_manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
    manifest_jobs = manifest.get("jobs", [])
    if not summaries and not manifest_jobs:
        return ""

    def source_questions_for(train_file: Any) -> list[dict[str, Any]] | None:
        if not isinstance(train_file, str):
            return None
        training_path = tier / "train" / Path(train_file).name
        canonical_path = tier / "train" / "train.jsonl"
        if not training_path.is_file() or not canonical_path.is_file():
            return None
        try:
            return sft_fact_questions(load_jsonl(training_path), load_jsonl(canonical_path))
        except ValueError:
            return None

    out = ["| Model label | training file | context / slot | validation fact overlap | expected questions | mean/p90 response latency (s) | seen-fact pass/weighted (answered/judged/expected) | held-out abstention pass/weighted (answered/judged/expected) | AI judge(s) | judge errors | invalid verdicts | unverified verdicts | exact source records |",
           "|---|---|---:|---|---:|---:|---|---|---|---:|---:|---:|---|"]
    completed_labels: set[str] = set()
    for summary_path in summaries:
        metadata = json.loads(summary_path.read_text())
        if metadata.get("schema") != "quillmoor-sft-facts-v1":
            continue
        train_file = metadata.get("train_file")
        expected_ids = metadata.get("expected_ids") or []
        expected = set(expected_ids)
        overlap = validation_fact_leakage(tier, train_file if isinstance(train_file, str) else None)
        source_questions = source_questions_for(train_file)
        source_ids = {question["id"] for question in source_questions} if source_questions is not None else None
        source_records = {
            question["id"]: (question["fact"], question["question"], question["answer"], question["type"])
            for question in source_questions or []
        }
        source_trained = {question["id"] for question in source_questions or [] if question["type"] == "sft-trained"}
        source_heldout = {question["id"] for question in source_questions or [] if question["type"] == "sft-heldout"}
        answer_files = metadata.get("answer_files") or []
        for answer_name in answer_files:
            answer_path = results / Path(answer_name).name
            completed_labels.add(answer_path.stem.removesuffix("__closed"))
            rows = load_jsonl(answer_path) if answer_path.is_file() else []
            actual = {str(row.get("id")) for row in rows if row.get("id") is not None}
            actual_records = {
                str(row["id"]): (row.get("fact"), row.get("question"), row.get("key"), row.get("type"))
                for row in rows if row.get("id") is not None
            }
            id_counts: dict[str, int] = {}
            for row in rows:
                if row.get("id") is not None:
                    row_id = str(row["id"])
                    id_counts[row_id] = id_counts.get(row_id, 0) + 1
            verified_rows = [
                row for row in rows
                if row.get("id") is not None
                and id_counts.get(str(row["id"])) == 1
                and str(row["id"]) in source_records
                and (row.get("fact"), row.get("question"), row.get("key"), row.get("type"))
                == source_records[str(row["id"])]
            ]
            expected_count = len(source_ids) if source_ids is not None else int(metadata.get("expected_questions") or len(expected))
            metadata_matches_source = (
                source_ids is not None
                and len(expected_ids) == len(expected)
                and expected == source_ids
                and int(metadata.get("expected_questions") or len(expected)) == len(source_ids)
                and int(metadata.get("expected_trained_facts") or 0) == len(source_trained)
                and int(metadata.get("expected_heldout_facts") or 0) == len(source_heldout)
                and int(metadata.get("train_sample_seed") or -1) == TRAIN_SAMPLE_SEED
            )
            exact_ids = (
                metadata_matches_source
                and len(rows) == len(expected)
                and actual == expected
                and actual_records == source_records
            )
            latency = response_latency(verified_rows)
            trained = score([row for row in verified_rows if row.get("type") == "sft-trained"])
            heldout = score([row for row in verified_rows if row.get("type") == "sft-heldout"])
            trained_expected = len(source_trained) if source_ids is not None else int(metadata.get("expected_trained_facts") or 0)
            heldout_expected = len(source_heldout) if source_ids is not None else int(metadata.get("expected_heldout_facts") or 0)

            def score_cell(metrics: dict[str, Any], expected_rows: int) -> str:
                exact = metrics.get("pass")
                weighted = metrics.get("score")
                exact_text = "—" if exact is None else f"{100 * exact:.1f}%"
                weighted_text = "—" if weighted is None else f"{100 * weighted:.1f}%"
                return f"{exact_text}/{weighted_text} ({metrics['answered']}/{metrics['judged']}/{expected_rows})"

            label = answer_path.stem.removesuffix("__closed")
            judge_models = sorted({str(row.get("judge_model") or "unrecorded") for row in verified_rows})
            judge_model_text = ", ".join(judge_models) if judge_models else "—"
            judge_errors = sum(bool(row.get("judge_error")) for row in verified_rows)
            valid_verdicts = {"pass", "partial", "fail"}
            invalid_verdicts = sum(
                bool(row.get("verdict"))
                and (not isinstance(row.get("verdict"), str) or row.get("verdict") not in valid_verdicts)
                for row in verified_rows
            )
            unverified_verdicts = sum(
                bool(row.get("verdict")) and not has_valid_judgment(row)
                for row in verified_rows
            )
            context_per_slot = metadata.get("ctx_per_slot") or 6144
            out.append(f"| `{label}` | `{train_file or '—'}` | {context_per_slot} | {overlap} | {expected_count} | {latency} | "
                       f"{score_cell(trained, trained_expected)} | {score_cell(heldout, heldout_expected)} | "
                       f"{judge_model_text} | {judge_errors} | {invalid_verdicts} | {unverified_verdicts} | "
                       f"{'yes' if exact_ids else 'no'} |")

    incomplete_jobs = [
        job for job in manifest_jobs
        if isinstance(job, dict) and str(job.get("label") or "") not in completed_labels
    ]
    if incomplete_jobs:
        out.extend([
            "", "**Incomplete evaluations (no score reported):**",
            "", "| Run | status | context / slot | answered / expected | judged / expected | answer errors | log |",
            "|---|---|---:|---:|---:|---:|---|",
        ])
        for job in incomplete_jobs:
            label = str(job.get("label") or "unknown").replace("|", "/")
            source_questions = source_questions_for(job.get("train_file"))
            source_records = {
                question["id"]: (question["fact"], question["question"], question["answer"], question["type"])
                for question in source_questions or []
            }
            source_ids = set(source_records)
            answer_path = results / f"{label}__closed.jsonl"
            rows = load_jsonl(answer_path) if answer_path.is_file() else []
            counts: dict[str, int] = defaultdict(int)
            for row in rows:
                if row.get("id") is not None:
                    counts[str(row["id"])] += 1
            verified_rows = [
                row for row in rows
                if row.get("id") is not None
                and counts.get(str(row["id"])) == 1
                and str(row["id"]) in source_ids
                and (row.get("fact"), row.get("question"), row.get("key"), row.get("type"))
                == source_records[str(row["id"])]
            ]
            expected_count = len(source_records) if source_questions is not None else None
            answered = sum(not row.get("error") and bool(str(row.get("answer") or "").strip())
                           for row in verified_rows)
            judged = sum(has_valid_judgment(row) for row in verified_rows)
            errors = sum(bool(row.get("error")) for row in verified_rows)
            exit_code = job.get("return_code")
            status = f"incomplete (exit {exit_code})" if exit_code not in (None, 0) else "no complete summary"
            context_per_slot = job.get("ctx_per_slot") or 6144
            answered_text = f"{answered}/{expected_count}" if expected_count is not None else str(answered)
            judged_text = f"{judged}/{expected_count}" if expected_count is not None else str(judged)
            log = str(job.get("log") or "—").replace("|", "/")
            out.append(f"| `{label}` | {status} | {context_per_slot} | {answered_text} | {judged_text} | {errors} | `{log}` |")

    if len(out) == 2:
        return ""
    out.extend(["", "Each model gets a deterministic, equal-sized sample of canonical facts it trained on and all canonical facts held out by the app's SFT split. Each score cell shows exact pass / weighted score (pass = 1, partial = 0.5, fail = 0), then answered / judged / expected. Response latency is mean / nearest-rank p90 per successful answer call, including retries but excluding model startup and judge time. The first cohort checks fact recall; the second expects abstention because those fictional facts are unknown to the closed-book model. AI judge identity, judge errors, and unrecognized non-empty verdicts are reported per model; scores from different judges are not directly comparable. The report recomputes both cohorts from the source dataset, checks the split has zero fact overlap, and requires exact question IDs, prompt text, answer keys, and fact IDs. Strong seen-fact recall plus strong held-out abstention is the desired result; use Smart Brain/RAG when answers must come from unseen documents."])
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiers", nargs="+", required=True)
    a = ap.parse_args()
    tiers = [Path(t) for t in a.tiers]
    print("## Ingest per file type\n\n" + ingest(tiers) + "\n")
    archetype_report = archetypes(tiers)
    if archetype_report:
        print("## Ingest by document archetype\n\n" + archetype_report + "\n")
    for t in tiers:
        for title, fn in (("Smart Brain retrieval", brain), ("Smart Brain export combinations", export_matrix),
                          ("Answers (AI judge)", answers), ("Fine-tuning runs", finetune),
                          ("SFT seen vs held-out fact answers (AI judge)", sft_facts),
                          ("Trained vs held-out facts", held_out)):
            body = fn(t)
            if body:
                print(f"## {title} — tier {t.name.upper()}\n\n{body}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
