"""Track A (training) browser run on the Korvane corpus: real UI, real files, no trust in any intermediate step.

    FTS_ALLOW_LIVE_E2E=1 .venv/bin/python tests/e2e_track_a.py --phase create --phase a_upload --phase a_prep ...
    FTS_ALLOW_LIVE_E2E=1 .venv/bin/python tests/e2e_track_a.py --list

Reuses the walkthrough harness (monitor, screenshots, journal watch) and adds phases whose gates are FACT coverage against the
corpus manifests (``tests/corpus/korvane``), not "at least one pair": ``a_upload`` uploads the tier's files and checks that every
manifest fact survived parsing; ``a_prep`` mines with the helper and checks that every fact reached an answer. Review is done by a
human (see ``scripts/corpus_review.py``), never by an "approve all" click.  Env: ``FTS_TIER`` (core | extended | all).
"""
from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import e2e_user_walkthrough as W
from corpus_build import ROOT, parse_source

W.PROJECT_NAME = os.environ.get("FTS_PROJECT_NAME", "korvane-core-1")
TIER = os.environ.get("FTS_TIER", "core")
LANES = [x for x in os.environ.get("FTS_LANES", "").split(",") if x]   # empty = every lane
FILES = [x for x in os.environ.get("FTS_FILES", "").split(",") if x]   # empty = every file of the tier
PY = str(W.REPO / ".venv" / "bin" / "python")


def tier_files() -> list[Path]:
    out = []
    for src in sorted((ROOT / "src").glob("*/*.src")):
        if LANES and src.parent.name not in LANES:
            continue
        meta, _ = parse_source(src)
        if FILES and meta["out"] not in FILES:
            continue
        if TIER == "all" or meta.get("tier", "core") == TIER:
            built = ROOT / "files" / src.parent.name / meta["out"]
            if not built.exists():
                raise SystemExit(f"{built} missing: run scripts/corpus_build.py --all")
            out.append(built)
    return out


def script(name: str, *args: str) -> tuple[int, str]:
    r = subprocess.run([PY, str(W.REPO / "scripts" / name), *args], capture_output=True, text=True, check=False,
                       env={**os.environ, "FTS_ROOT": os.environ.get("FTS_ROOT", str(Path.home() / ".finetune-studio"))})
    return r.returncode, r.stdout + r.stderr


async def phase_a_upload(w: W.Walk) -> None:
    """File library: UPLOAD every file of the tier, wait for parsing, then verify no manifest fact was lost by a parser."""
    files = tier_files()
    W.log(f"uploading {len(files)} {TIER} files")
    await w.goto(f"/projects/{W.pid()}/data")
    await w.page.click("#fb-upload-toggle")
    await w.page.set_input_files("#fb-upload-input", [str(f) for f in files])
    await w.shot("files-chosen")
    await w.page.click("#fb-upload-form button[type=submit]")

    async def parsed() -> bool:
        srcs = W.api(f"/api/projects/{W.pid()}/data-prep/sources").get("sources", [])
        return len(srcs) >= len(files) and all(s.get("status") == "ready" for s in srcs)

    W.R.check(await w.wait_for("every file parsed", parsed, 900, every=5), f"all {len(files)} files parsed")
    await w.page.reload(wait_until="networkidle")
    await w.shot("library", full=True)
    code, out = script("corpus_parse_check.py", "--pid", W.pid(), "--tier", TIER, *(["--lanes", *LANES] if LANES else []), *(["--files", *FILES] if FILES else []))
    W.log("parse check:\n" + out[-6000:])
    W.R.check(code == 0, "parsers kept every manifest fact (corpus_parse_check)")


async def phase_a_prep(w: W.Walk) -> None:
    """Pairs page: GENERATE PAIRS FOR SELECTED on every source, then measure fact coverage (the gate)."""
    await w.goto(f"/projects/{W.pid()}/data-prep")
    n = await W._select_all_parsed_sources(w)
    W.R.check(n > 0, f"selected {n} parsed sources")
    await w.page.click("#dp-generate-selected")
    await w.page.wait_for_timeout(2500)
    started = time.time()

    async def finished() -> bool:
        srcs = W.api(f"/api/projects/{W.pid()}/data-prep/sources")["sources"]
        return all(x["status"] in ("generated", "generated_incomplete", "error", "failed") for x in srcs)

    W.R.check(await w.wait_for("every source mined", finished, 3600, every=15), "helper finished every source")
    W.log(f"mining took {time.time() - started:.0f}s for {n} sources")
    await w.page.click("button:has-text('REFRESH') >> nth=0")
    await w.page.wait_for_timeout(1500)
    await w.shot("generation-finished", full=True)
    for x in W.api(f"/api/projects/{W.pid()}/data-prep/sources")["sources"]:
        W.log(f"  {x['filename']:52s} {x['status']:20s} pairs={x['pairs_total']:4d} chunks={x.get('chunk_count')}")
    code, out = script("corpus_coverage.py", "--pid", W.pid(), "--tier", TIER, "--status", "any", "--json",
                       str(W.WORK / "coverage-mined.json"), *(["--lanes", *LANES] if LANES else []), *(["--files", *FILES] if FILES else []))
    W.log("coverage:\n" + out[-9000:])
    W.R.check(code == 0, "every manifest fact is covered by a mined pair (100 %)")


W.PHASES.update({"a_upload": phase_a_upload, "a_prep": phase_a_prep})

async def phase_a_export(w: W.Walk) -> None:
    """Pairs page: EXPORT APPROVED -> TRAINING with plain rows (no retrieved context: RAG is a separate track)."""
    await w.goto(f"/projects/{W.pid()}/data-prep")
    grounded = w.page.locator("#dp-grounded")
    if await grounded.is_checked():
        await grounded.uncheck()
    await w.shot("export-options")
    await w.page.click("#dp-export-approved")
    await w.page.wait_for_timeout(5000)
    await w.shot("export-clicked", full=True)
    ds = W.api(f"/api/projects/{W.pid()}/datasets").get("datasets", [])
    W.R.check(len(ds) >= 1, f"a dataset was registered ({[d['name'] for d in ds]})")
    if ds:
        W.log(f"dataset: {ds[0]['name']} rows={ds[0]['qa_count']}")
        W.R.check("retrieved context" not in ds[0]["name"] or "0 with retrieved context" in ds[0]["name"],
                  "the training dataset carries plain rows only")


async def phase_a_train(w: W.Walk) -> None:
    """Training page: Qwen3.5-9B + this project's dataset + explicit SFT settings; watch the live panel; record the loss curve."""
    epochs = os.environ.get("FTS_EPOCHS", "3")
    lr = os.environ.get("FTS_LR", "1e-4")
    await w.goto(f"/projects/{W.pid()}/training")
    base = w.page.locator("#train-base-model")
    opts = await base.locator("option").all_inner_texts()
    choice = next((o for o in opts if "Qwen3.5-9B" in o), None)
    W.R.check(choice is not None, f"base model dropdown offers Qwen3.5-9B ({len(opts)} models)")
    await base.select_option(label=choice)
    await w.page.click("button:has-text('From this project')")
    ds = w.page.locator("#dataset-select")
    dopts = await ds.locator("option").all_inner_texts()
    pick = next((o for o in dopts if "sharegpt" in o or "approved" in o.lower()), dopts[-1])
    await ds.select_option(label=pick)
    for name, value in (("lora_rank", "64"), ("learning_rate", lr), ("num_epochs", epochs), ("batch_size", "2"),
                        ("max_seq_length", "1024")):
        await w.page.fill(f"input[name={name}]", value)
    await w.page.fill("#gradient-accum-steps", "8")
    await w.page.fill("#warmup-steps", "20")
    await w.page.fill("#eval-steps", os.environ.get("FTS_EVAL_STEPS", "25"))
    want_early = os.environ.get("FTS_EARLY_STOP", "1") != "0"
    if await w.page.locator("#early-stopping-check").is_checked() != want_early:
        await (w.page.locator("#early-stopping-check").check() if want_early else w.page.locator("#early-stopping-check").uncheck())
    if await w.page.locator("#merge-on-save-check").is_checked() and os.environ.get("FTS_MERGE", "1") == "0":
        await w.page.locator("#merge-on-save-check").uncheck()
    await w.shot("configured", full=True)
    await w.page.click("#start-btn")
    await w.page.wait_for_timeout(5000)
    await w.shot("training-started", full=True)
    last = ""
    started = time.time()
    while time.time() - started < float(os.environ.get("FTS_TRAIN_TIMEOUT", "7200")):
        st = W.api("/api/training/status")
        line = (f"training {st.get('status')} step {st.get('step')}/{st.get('total_steps')} loss={st.get('loss')} "
                f"eta={st.get('eta')} | {W.gpu_used_mib()}")
        if line != last:
            W.log(line)
            last = line
        if st.get("status") in ("done", "error", "stopped", "idle") and time.time() - started > 30:
            break
        await asyncio.sleep(30)
    st = W.api("/api/training/status")
    await w.shot("training-finished", full=True)
    W.R.check(st.get("status") == "done", f"training finished: {st.get('status')} step {st.get('step')}/{st.get('total_steps')} "
              f"loss {st.get('loss')} {str(st.get('error') or '')[:200]}")
    runs = W.api(f"/api/training/runs/{W.pid()}")
    if runs:
        W.save_state(run_id=runs[0]["id"], run_path=runs[0].get("output_path", ""))
        W.log("run record: " + json.dumps({k: runs[0].get(k) for k in ("id", "status", "final_loss", "eval_loss", "metrics_json")},
                                          default=str)[:1200])


W.PHASES.update({"a_export": phase_a_export, "a_train": phase_a_train})




async def phase_a_chat(w: W.Walk) -> None:
    """Chat page: LOAD the exported q4_k_m through the UI, then quiz it with the corpus eval (paraphrase recall + abstention)."""
    W.api("/api/models/unload", "POST")
    await w.goto(f"/projects/{W.pid()}/chat")

    async def options_ready() -> bool:
        return await w.page.locator("#chat-inline-model option").count() > 0

    await w.wait_for("Chat model choices", options_ready, 20, every=0.5)
    options = await w.page.locator("#chat-inline-model option").evaluate_all(
        "els => els.map(e => ({value: e.value, text: e.textContent}))")
    run_id = os.environ.get("FTS_RUN_ID", "")
    pick = next((o["value"] for o in options if "q4_k_m" in o["value"].lower() and (not run_id or f"/{run_id}/" in o["value"])), None)
    W.R.check(pick is not None, f"the exported q4_k_m model of run {run_id or 'any'} is selectable in Chat ({len(options)} options)")
    if pick is None:
        return
    await w.page.select_option("#chat-inline-model", value=pick)
    await w.page.click("#chat-inline-load-btn")

    async def loaded() -> bool:
        return bool(W.api("/api/inference/status").get("loaded"))

    W.R.check(await w.wait_for("model loaded in Chat", loaded, 180, every=2), "model loaded")
    await w.shot("loaded")
    st = W.api("/api/inference/status")
    W.log(f"placement: ctx={st.get('n_ctx')} layers={st.get('n_gpu_layers')} offload={st.get('offload')}")
    out = W.REPO / ".tmp" / "eval"
    out.mkdir(parents=True, exist_ok=True)
    tag = os.environ.get("FTS_EVAL_TAG", "run")
    rc, text = script("corpus_eval.py", "--json", str(out / f"{tag}.json"))
    W.log(f"corpus_eval rc={rc}\n" + text[-1500:])
    await w.page.fill("#chat-input", "What incident reference number did Korvane open for the Oakhaven temperature excursion?")
    await w.page.click("#chat-send")
    await w.page.wait_for_timeout(15000)
    await w.shot("chat-answer", full=True)


W.PHASES["a_chat"] = phase_a_chat


async def phase_a_dpo_build(w: W.Walk) -> None:
    """Pairs page: Preference pairs (DPO) card -> BUILD with the local helper (hallucination + abstain), read the summary."""
    W.api("/api/models/unload", "POST")
    W.api("/api/providers/local-default/load", "POST", {"n_ctx": 32768})
    await w.goto(f"/projects/{W.pid()}/data-prep")
    await w.page.fill("#dp-pref-max", os.environ.get("FTS_DPO_PAIRS", "300"))
    await w.shot("dpo-card", full=True)
    await w.page.click("#dp-pref-build")
    started = time.time()
    last = ""
    while time.time() - started < float(os.environ.get("FTS_DPO_TIMEOUT", "3600")):
        text = (await w.page.locator("#dp-pref-progress-text").inner_text()).strip()
        if text != last:
            W.log(f"preference build: {text}")
            last = text
        if await w.page.locator("#dp-pref-result").is_visible():
            break
        status = (await w.page.locator("#dp-pref-status").inner_text()).strip() if await w.page.locator("#dp-pref-status").is_visible() else ""
        if status:
            W.log(f"preference status: {status[:400]}")
            break
        await asyncio.sleep(10)
    await w.shot("dpo-built", full=True)
    result = (await w.page.locator("#dp-pref-result").inner_text()).strip() if await w.page.locator("#dp-pref-result").is_visible() else ""
    W.log("preference result: " + result.replace("\n", " | ")[:900])
    W.R.check(bool(result), "the preference pairs were built and a Train-with-this-dataset link is offered")


W.PHASES["a_dpo_build"] = phase_a_dpo_build


async def phase_a_dpo_train(w: W.Walk) -> None:
    """Training page: Preference tuning (DPO) from the SFT run's merged model, with the REVIEWED preference JSONL uploaded."""
    W.api("/api/providers/unload", "POST")
    run_id = os.environ.get("FTS_RUN_ID", "")
    await w.goto(f"/projects/{W.pid()}/training")
    await w.page.check("input[name=training_mode][value=dpo]")
    base = w.page.locator("#train-base-model")
    opts = await base.locator("option").all_inner_texts()
    await base.select_option(label=next(o for o in opts if "Qwen3.5-9B" in o))
    start = w.page.locator("#start-run-select")
    await w.wait_for("SFT runs offered as a DPO start", lambda: _has_options(start), 20, every=1)
    values = await start.locator("option").evaluate_all("els => els.map(e => e.value)")
    W.log(f"start-from options: {values}")
    W.R.check(run_id in values, f"the SFT run {run_id} is offered as the DPO starting point")
    await start.select_option(value=run_id)
    await w.page.click("button.data-tab[data-tab=upload]")
    await w.page.set_input_files("#dataset-upload", str(W.REPO / ".tmp" / "reviewed-preference.jsonl"))
    await w.page.wait_for_timeout(4000)
    W.log("upload status: " + (await w.page.locator("#dataset-upload-status").inner_text()).strip()[:300])
    await w.shot("dpo-configured", full=True)
    await w.page.click("#start-btn")
    await w.page.wait_for_timeout(5000)
    last = ""
    started = time.time()
    while time.time() - started < float(os.environ.get("FTS_TRAIN_TIMEOUT", "7200")):
        st = W.api("/api/training/status")
        line = f"dpo {st.get('status')} step {st.get('step')}/{st.get('total_steps')} loss={st.get('loss')} {str(st.get('message'))[:80]}"
        if line != last:
            W.log(line)
            last = line
        if st.get("status") in ("done", "error", "stopped", "idle") and time.time() - started > 30:
            break
        await asyncio.sleep(20)
    st = W.api("/api/training/status")
    await w.shot("dpo-finished", full=True)
    W.R.check(st.get("status") == "done", f"DPO finished: {st.get('status')} {str(st.get('error') or '')[:300]}")
    runs = W.api(f"/api/training/runs/{W.pid()}")
    if runs:
        W.log("run record: " + json.dumps({k: runs[0].get(k) for k in ("id", "status", "final_loss", "settings")}, default=str)[:900])


async def _has_options(locator) -> bool:
    return await locator.locator("option").count() > 1


W.PHASES["a_dpo_train"] = phase_a_dpo_train


def _retrieval_quiz(pid: str) -> tuple[int, int]:
    """How many paraphrase questions have every expected value in the top-5 RAG chunks (blocking HTTP: run in a thread)."""
    import urllib.request as _u

    from corpus_coverage import norm
    from corpus_eval import variants

    rows = [json.loads(ln) for ln in (ROOT / "eval" / "paraphrase_core.jsonl").read_text().splitlines() if ln.strip()]
    hit = 0
    for r in rows:
        req = _u.Request(f"{W.BASE}/api/projects/{pid}/rag/search", method="POST", headers={"Content-Type": "application/json"},
                         data=json.dumps({"query": r["q"], "top_k": 5}).encode())
        with _u.urlopen(req, timeout=120) as resp:
            hits = json.loads(resp.read()).get("hits", [])
        blob = norm(" ".join(str(h.get("text", "")) for h in hits))
        hit += all(any(v in blob or v in blob.replace(",", "") for v in variants(x)) for x in r["expect"])
    return hit, len(rows)


async def phase_b_rag(w: W.Walk) -> None:
    """Track B (RAG, separate from training): QUICK INDEX the project's files, SEARCH like a user, then quiz RAG chat with the loaded model."""
    W.api("/api/models/unload", "POST")
    await w.goto(f"/projects/{W.pid()}/rag")
    await w.shot("rag-before")
    await w.page.click("#quick-index-btn")

    async def built() -> bool:
        try:
            d = W.api(f"/api/projects/{W.pid()}/rag/build/status")
            return bool(d.get("ok", True)) and not d.get("building") and int(d.get("chunks") or d.get("chunk_count") or 0) > 0
        except Exception:  # noqa: BLE001
            return False

    W.R.check(await w.wait_for("RAG index built", built, 900, every=5), "quick index finished")
    st = W.api(f"/api/projects/{W.pid()}/rag")
    W.log("rag status: " + json.dumps(st, default=str)[:500])
    await w.page.reload(wait_until="networkidle")
    await w.shot("rag-built", full=True)
    for question in ("What incident reference number did Korvane open for the Oakhaven temperature excursion?",
                     "Which certificate number did the lead auditor recommend maintaining?",
                     "What is the Autumn Peak surcharge in the Oakhaven MSA?"):
        await w.page.fill("#q-text", question)
        await w.page.click("#q-btn")
        await w.page.wait_for_timeout(4000)
        await w.shot("rag-search")
        W.log(f"searched: {question}")
    # retrieval-only quiz over every paraphrase question (no model involved)
    hit, total = await asyncio.to_thread(_retrieval_quiz, W.pid())
    rows = range(total)
    W.log(f"retrieval@5 over {len(rows)} paraphrase questions: {hit}/{len(rows)} = {100 * hit / len(rows):.1f}%")
    W.R.check(hit >= int(0.8 * len(rows)), f"RAG search surfaced the expected facts for {hit}/{len(rows)} questions (>= 80 %)")


W.PHASES["b_rag"] = phase_b_rag


async def phase_a_guide(w: W.Walk) -> None:
    """Guide (Agent mode, the in-app assistant) answers a question about this project through the configured helper."""
    W.api("/api/models/unload", "POST")
    W.api("/api/providers/local-default/load", "POST", {"n_ctx": 32768})
    await w.goto(f"/projects/{W.pid()}")
    await w.page.click("button:has-text('guide')")
    await w.page.wait_for_selector("#guide-panel:not([hidden])", timeout=10000)
    await w.page.fill("#guide-input", "Where does this project stand, and what should I do next? Be specific about this project's runs and datasets.")
    await w.page.click("#guide-send")

    async def answered() -> bool:
        return not await w.page.locator("#guide-stop").is_enabled()

    await w.page.wait_for_timeout(5000)
    W.R.check(await w.wait_for("Guide answer", answered, 240, every=3), "the Guide finished answering")
    text = (await w.page.locator("#guide-msgs").inner_text()).strip()
    W.log("guide transcript: " + text.replace("\n", " | ")[:1200])
    await w.shot("guide-answer", full=True)
    W.R.check(len(text) > 200, "the Guide produced a substantive answer")


W.PHASES["a_guide"] = phase_a_guide


async def phase_a_cleanup(w: W.Walk) -> None:
    """Project overview: DELETE the project like a user; its runs, exports, datasets and RAG corpus must be gone, the base model must stay."""
    p = W.pid()
    W.api("/api/models/unload", "POST")
    await w.goto(f"/projects/{p}")
    await w.shot("before-delete")
    await w.page.click("button:has-text('DELETE')")
    await w.page.wait_for_timeout(600)
    await w.page.click("button:has-text('OK')")
    await w.page.wait_for_timeout(4000)
    W.R.check(not [x for x in W.api("/api/projects") if x["id"] == p], "project is gone from the API")
    leftovers = [str(d) for d in (W.REPO / "output" / "projects" / p, W.REPO / "data" / "projects" / p,
                                  Path.home() / ".finetune-studio" / "projects" / p,
                                  Path.home() / ".finetune-studio" / "rag_corpora" / p) if d.exists()]
    W.R.check(not leftovers, f"no project directories left on disk {leftovers}")
    W.R.check((W.REPO / "models" / "safetensors" / "Qwen3.5-9B" / "config.json").exists(), "the shared Qwen3.5-9B base model is still there")


W.PHASES["a_cleanup"] = phase_a_cleanup


# ── UI-only review and scoring ─────────────────────────────────────────────────────────────────────────────────────────────────
REASONS = ["wrong value", "quote echo / not a real question", "duplicate of another pair", "junk / unusable",
           "missing unit or wrong premise"]  # same order as the Pairs page's reason chips (keys 1-5)


async def _review_state(w: W.Walk) -> dict | None:
    return await w.page.evaluate(
        "(() => { const it = _rq.items[_rq.cur]; return it ? {id: it.id, file: it.source_filename, chunk: it.chunk_idx, "
        "q: it.question, a: it.answer, total: _rq.total, counts: _rq.counts} : null; })()")


async def _settled(w: W.Walk) -> None:
    await w.page.wait_for_function("!_rq.loading && _rq.inflight === 0", timeout=60000)
    await w.page.evaluate("document.activeElement && document.activeElement.blur && document.activeElement.blur()")


async def _next_pair(w: W.Walk, before_id: str) -> None:
    await w.page.wait_for_function("id => (_rq.items[_rq.cur] || {}).id !== id", arg=before_id, timeout=15000)


async def phase_a_review_ui(w: W.Walk) -> None:
    """Pairs page: replay the reader's verdicts with the keyboard (A / R+reason / edit+Ctrl+Enter / N add). A pair that has no
    verdict stops the run: nothing is ever approved by default and there is no 'approve all' click."""
    path = Path(os.environ.get("FTS_VERDICTS", str(W.REPO / ".tmp" / "ui-verdicts.jsonl")))
    verdicts: dict[str, dict] = {}
    adds: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("add"):
            adds.append(row)
        else:
            verdicts[row["id"]] = row
    W.log(f"replaying {len(verdicts)} verdicts and {len(adds)} additions from {path.name}")
    await w.goto(f"/projects/{W.pid()}/data-prep")
    await w.page.wait_for_selector("#dp-rq-detail textarea#dp-rq-q", timeout=60000)
    await w.page.click("#dp-jump-review")
    await w.page.wait_for_timeout(800)
    await w.shot("review-start")
    done, started, missing = 0, time.time(), None
    pending_adds = list(adds)
    while True:
        await _settled(w)
        st = await _review_state(w)
        if st is None:
            break
        for add in [x for x in pending_adds if x["file"] == st["file"] and int(x["chunk"]) == int(st["chunk"] or 0)]:
            if not await w.page.locator("#dp-rq-add").is_visible():
                await w.page.keyboard.press("n")
            await w.page.fill("#dp-rq-add-q", add["q"])
            await w.page.fill("#dp-rq-add-a", add["a"])
            await w.page.press("#dp-rq-add-a", "Control+Enter")
            await _settled(w)
            pending_adds.remove(add)
        st = await _review_state(w)
        if st is None:
            break
        v = verdicts.get(st["id"])
        if v is None:
            missing = st
            break
        kind = v["v"]
        if kind == "A":
            await w.page.keyboard.press("a")
        elif kind == "E":
            await w.page.fill("#dp-rq-q", v["q"])
            await w.page.fill("#dp-rq-a", v["a"])
            await w.page.press("#dp-rq-a", "Control+Enter")
        else:
            reason = (v.get("reason") or "").strip() or "junk / unusable"
            await w.page.keyboard.press("r")
            if reason in REASONS:
                await w.page.keyboard.press(str(REASONS.index(reason) + 1))
            else:
                await w.page.keyboard.press("0")
                await w.page.keyboard.type(reason)
                await w.page.keyboard.press("Enter")
        await _next_pair(w, st["id"])
        done += 1
        if done % 200 == 0:
            W.log(f"reviewed {done} pairs in {time.time() - started:.0f}s; pending {st['counts']['pending']}")
    await _settled(w)
    await w.shot("review-end", full=True)
    counts = W.api(f"/api/projects/{W.pid()}/data-prep/qa/queue?limit=1")["counts"]
    W.log(f"replayed {done} verdicts + {len(adds) - len(pending_adds)} additions in {time.time() - started:.0f}s; server counts {counts}")
    W.R.check(missing is None, "every pending pair had a reader's verdict" if missing is None else
              f"STOPPED: pair {missing['id']} ({missing['file']} c{missing['chunk']}) has no verdict; {missing['counts']['pending']} still pending")
    W.R.check(not pending_adds, f"all {len(adds)} reviewer-written pairs were added in the UI ({len(pending_adds)} left)")
    if missing is None:
        W.R.check(counts["pending"] == 0, "nothing is left pending")
    pairs = W.api(f"/api/projects/{W.pid()}/data-prep/qa")["items"]
    no_stamp = [p for p in pairs if p["status"] in ("approved", "rejected") and not p.get("reviewed_at")]
    W.R.check(not no_stamp, f"every approved/rejected pair carries a human verdict stamp ({len(no_stamp)} without)")
    rejected_no_reason = [p for p in pairs if p["status"] == "rejected" and not p.get("note")]
    W.R.check(not rejected_no_reason, f"every rejected pair carries a reason ({len(rejected_no_reason)} without)")


async def _quiz_results(w: W.Walk, tag: str) -> dict:
    """Read the finished run off the Testing page itself (summary line + every case row) and keep it as JSON."""
    summary = (await w.page.inner_text("#case-scores-summary")).strip()
    cats = (await w.page.inner_text("#case-scores-categories")).strip() if await w.page.locator("#case-scores-categories").count() else ""
    rows = await w.page.locator("#case-results-table tbody tr").evaluate_all(
        "trs => trs.map(tr => ({name: tr.querySelector('.case-col-name').textContent.trim(), "
        "category: tr.querySelector('.case-col-cat').textContent.trim(), "
        "question: tr.querySelector('.case-col-text').textContent.trim(), "
        "answer: tr.querySelector('.case-answer-pre').textContent, verdict: tr.querySelector('.verdict-badge').textContent.trim(), "
        "why: tr.querySelector('.judge-reasoning').textContent.trim()}))")
    out = W.REPO / ".tmp" / "ui-results"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{tag}.json").write_text(json.dumps({"summary": summary, "categories": cats, "cases": rows}, indent=1, ensure_ascii=False))
    by: dict[str, dict[str, int]] = {}
    for r in rows:
        c = by.setdefault(r["category"], {"pass": 0, "partial": 0, "fail": 0, "none": 0})
        c[r["verdict"] if r["verdict"] in c else "none"] += 1
    W.log(f"[{tag}] {summary} | {cats}")
    for cat, c in by.items():
        total = sum(c.values())
        W.log(f"[{tag}] {cat}: pass {c['pass']}/{total} = {100 * c['pass'] / max(total, 1):.1f}%  partial {c['partial']}  fail {c['fail']}")
    return by


async def phase_a_test_ui(w: W.Walk) -> None:
    """Testing page: IMPORT the Korvane quiz, pick a model, RUN, read the scores off the page. FTS_TEST_MODEL = substring of the
    model option (path or label); empty = the page's own default. FTS_EVAL_TAG names the saved results."""
    tag = os.environ.get("FTS_EVAL_TAG", "run")
    want = os.environ.get("FTS_TEST_MODEL", "")
    await w.goto(f"/projects/{W.pid()}/testing")
    await w.page.click("#t-suite-import-box summary")
    await w.page.set_input_files("#t-suite-file", str(W.REPO / "tests" / "corpus" / "korvane" / "eval" / "korvane_quiz_core.jsonl"))
    await w.page.click("#t-suite-import-btn")
    await w.page.wait_for_function("document.getElementById('t-suite-import-msg').textContent.startsWith('Imported')", timeout=30000)
    W.log("quiz import: " + await w.text("#t-suite-import-msg"))
    if want:
        opts = await w.page.locator("#t-model option").evaluate_all("els => els.map(e => ({v: e.value, t: e.textContent}))")
        pick = next((o["v"] for o in opts if want.lower() in (o["v"] + " " + o["t"]).lower()), None)
        W.R.check(pick is not None, f"the Testing page offers a model matching {want!r} ({len(opts)} options)")
        if pick is None:
            return
        await w.page.select_option("#t-model", value=pick)
    W.api("/api/models/unload", "POST")
    await w.shot("quiz-ready", full=True)
    await w.page.click("#t-run-btn")
    started = time.time()

    async def done() -> bool:
        return await w.page.locator("#case-scores-summary").count() > 0

    ok = await w.wait_for("quiz results on the page", done, float(os.environ.get("FTS_QUIZ_TIMEOUT", "2400")), every=10)
    W.R.check(ok, f"the quiz finished in {time.time() - started:.0f}s")
    if ok:
        await w.shot("quiz-results", full=True)
        by = await _quiz_results(w, tag)
        W.R.check(sum(sum(c.values()) for c in by.values()) == 122, "all 122 questions were scored")


async def phase_a_eval_ui(w: W.Walk) -> None:
    """Testing page, Dataset evaluation card: FTS_EVAL_KIND = heldout | training_leakage (the memorization control)."""
    kind = os.environ.get("FTS_EVAL_KIND", "training_leakage")
    tag = os.environ.get("FTS_EVAL_TAG", kind)
    await w.goto(f"/projects/{W.pid()}/testing")
    want = os.environ.get("FTS_TEST_MODEL", "")
    if want:
        opts = await w.page.locator("#t-model option").evaluate_all("els => els.map(e => ({v: e.value, t: e.textContent}))")
        pick = next((o["v"] for o in opts if want.lower() in (o["v"] + " " + o["t"]).lower()), None)
        W.R.check(pick is not None, f"the Testing page offers a model matching {want!r}")
        if pick is None:
            return
        await w.page.select_option("#t-model", value=pick)
    await w.page.select_option("#t-eval-kind", value=kind)
    await w.page.fill("#t-train-max", os.environ.get("FTS_EVAL_MAX", "200"))
    W.api("/api/models/unload", "POST")
    await w.shot("eval-ready")
    await w.page.click("#t-train-eval-btn")

    async def done() -> bool:
        return await w.page.locator("#case-scores-summary").count() > 0

    ok = await w.wait_for(f"{kind} results", done, float(os.environ.get("FTS_QUIZ_TIMEOUT", "2400")), every=10)
    W.R.check(ok, f"the {kind} evaluation finished")
    if ok:
        await w.shot(f"{kind}-results", full=True)
        await _quiz_results(w, tag)


W.PHASES.update({"a_review_ui": phase_a_review_ui, "a_test_ui": phase_a_test_ui, "a_eval_ui": phase_a_eval_ui})


if __name__ == "__main__":
    raise SystemExit(W.main())
