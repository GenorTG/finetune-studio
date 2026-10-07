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
    if not await w.page.locator("#early-stopping-check").is_checked():
        await w.page.locator("#early-stopping-check").check()
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


if __name__ == "__main__":
    raise SystemExit(W.main())
