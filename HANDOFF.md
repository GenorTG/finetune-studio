# HANDOFF — finetune-studio

## Mission
Ships a local fine-tune + data-prep WebUI (FastAPI, `src/finetune_studio/`) with a data-prep tool-calling chat, training/merge/GGUF export, and an E2E browser test suite.
Genor edits on genorbox1 and pushes to GitHub; fan-dragon pulls and runs it as a bare `uvicorn` on port 7860 (RTX 3090). No systemd by default.

## State (verified 2026-09-14 13:05)
| Area | Status |
|------|--------|
| Git | `main` @ `4bebc37`, clean; fan-dragon pulled `4bebc37` at 13:04 (bare uvicorn pid 950380 on :7860) |
| F821 imports | ✅ all 17 fixed (`4c7b64c` + `497c9b7` revert-only). `ruff check src/ --select F821` → `All checks passed!`. `pytest tests/test_vram.py -q` → `17 passed in 0.05s` (was 16 fail). Cursor also restored `RAGStore.rebuild_vectors(embedder, device)` signature that commit `4632794` accidentally deleted — was raising `AttributeError` on `main`. |
| Ruff other | ⚠️ 609 findings remain (was 622; −17 F821, +4 pre-existing `UP045` ruff can now resolve). Out of scope per Cursor's task. |
| Data-prep layout | ✅ fixed in `4bebc37`. Playwright ancestor walk on fan-dragon identified `<main id="content" class="content">` as the clipping box (`flex:1 1 0` + `overflow:hidden auto` inside a 361px grid row). Removed the inner scroll cage on `.app`/`.main`/`.content`. File library (NAME/TYPE/SIZE/STATUS/ACTIONS, 8+ rows visible) now renders at 1280×800 and 1920×1080 — both screenshots viewed. |
| Deploy docs | ✅ `RESTART.md` fan-dragon section rewritten for the bare-uvicorn reality (start-then-kill-old pid pattern, `</dev/null` + `--no-access-log` + `bash -lc` for fish). `scripts/finetune-studio.service` + `scripts/install-service.sh` added for the optional systemd path. `AGENTS.md` Commands updated. |
| Header/tabs | ✅ `c1f8872` + `2d4951c` (`margin-right: 490px`) deployed; top-right cluster + scrollable tabs verified. |
| Models page | ✅ `/models` with category pills; trained exports resolved via project lookup (`aec11e6`..`25f3a3e`) |
| Data-prep chat | ✅ Implemented in `src/finetune_studio/webui/routes/data_prep_chat.py`; local model or external OpenAI-compatible API |
| Unload VRAM | ⚠️ Fixed in `a34c08b`; not regression-tested since 2026-09-10 |
| E2E suite (PHASES.md) | ⚠️ Stopped at Phase 3 (upload 10 fixtures to project `2026-09-10-oftest`, id `b08426e3`); Phases 3–8 pending |
| Old training runs | ⚠️ `bdc217b1` + 2 others still show empty `error` columns (failed before `daf9fa1`); backfill TODO |

## Next steps
1. Ruff auto-fix pass on the remaining 609 findings — `.venv/bin/python -m ruff check src/ --fix` (F401/I001/UP045/RUF100/F541 only, review diff) — done when findings drop below 100 and the test suite still passes (`pytest tests/ --ignore=tests/unit/test_vram_profiler.py --ignore=tests/test_breakpoints.py --ignore=tests/test_breakpoints_visual.py --ignore=tests/test_phase_bd.py --ignore=tests/test_phase_bd_api.py`)
2. Backfill `training_runs.error` for the 3 pre-`daf9fa1` failed runs from `/tmp/uvicorn.log` (or known messages) — done when `/projects/b08426e3/training?run=bdc217b1` shows the actual `'TrainingEngine' object has no attribute '_auto_generate_suite'` traceback
3. Resume E2E Phase 3 — `tests/run_qa.sh` (rules in `tests/README_E2E.md`); upload the 10 `OCTOPUS-7741` fixtures from `tests/fixtures/` to project `2026-09-10-oftest`, run the Qwen Q&A pipeline, export JSONL — done when the JSONL exists and every row contains `OCTOPUS-7741`
4. E2E Phase 4 — start training on the exported JSONL with a small base model and low epochs — done when the loaded Qwen model is unloaded before training starts and the run reaches 100% in the Training tab
5. Optional: actually run `bash scripts/install-service.sh` on fan-dragon and verify `systemctl status finetune-studio` shows active — done when systemd owns the port (replaces bare-uvicorn pid 950380)
6. Visual re-check of all 9 project tabs scrolling on one line at 1920×1080 (only 1280×800 + 1920×1080 confirmed for `/data-prep` in this session) — done when every project tab is reachable without horizontal-scroll mismatch
7. `HANDOFF.md` self-audit: confirm ≤120 lines after this rewrite (currently 64)

## Commands
- Test (genorbox1, pure): `.venv/bin/python -m pytest tests/ -q --tb=no -p no:cacheprovider --ignore=tests/unit/test_vram_profiler.py --ignore=tests/test_breakpoints.py --ignore=tests/test_breakpoints_visual.py --ignore=tests/test_phase_bd.py --ignore=tests/test_phase_bd_api.py`
- Test (fan-dragon, GPU + browser): `ssh fan-dragon 'cd /home/genortg/finetune-studio && .venv/bin/python -m pytest tests/ -q --tb=short -p no:cacheprovider'`
- Deploy: `git push`, then `ssh fan-dragon 'cd /home/genortg/finetune-studio && git pull --ff-only'`; restart bare uvicorn per `RESTART.md` "Start / restart the WebUI on :7860" (start new → kill old pid → `ss -ltnp | grep 7860`). Never `make run` on genorbox1 (dev-only box, no models/GPU)
- Verify (genorbox1): `.venv/bin/python -m ruff check src/ --select F821 && <Test (genorbox1) line>`; visual/E2E only on fan-dragon (`http://fan-dragon:7860/`)

## Blockers
- genorbox1 is dev-only (no install, no GPU, no models): pure unit tests + ruff here; GPU tests, Playwright tests, E2E Phases 3–8 and every visual check on fan-dragon after `git pull`
