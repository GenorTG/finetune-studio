# HANDOFF — finetune-studio

## Mission
Ships a local fine-tune + data-prep WebUI (FastAPI, `src/finetune_studio/`) with a data-prep tool-calling chat, training/merge/GGUF export, and an E2E browser test suite.
Genor edits on genorbox1 and pushes to GitHub; fan-dragon pulls and runs it as `finetune-studio.service` on port 7860 (RTX 3090).

## State (verified 2026-09-14 12:10)
| Area | Status |
|------|--------|
| Git | `main` @ `b988b2d`, clean; fan-dragon pulled `b988b2d` at 11:52 |
| Unit tests (genorbox1) | ⚠️ pytest + ruff installed 2026-09-14 (`uv pip install -e .[dev]`). Pure tests: **204 pass / 16 fail** — all 16 in `tests/test_vram.py` from one bug: `Path` not imported in `src/finetune_studio/training/vram/profile.py:28`. 4 files need playwright → fan-dragon only (`test_breakpoints*.py`, `test_phase_bd*.py`); `tests/unit/test_vram_profiler.py` needs GPU |
| Ruff | ❌ 622 findings; **17 × F821 undefined-name are real runtime `NameError`s on error paths**: `JSONResponse` in `webui/routes/{projects,benchmarks}.py`, `HTTPException` in `routes/pages.py:215`, `json` in `training/engine.py:603,637`, `Path`/`Optional` in `data/rag_portable/query.py` + `training/vram/profile.py`, `PortableRAG` in `data/rag_eval.py:177`, `embedder`/`device` in `rag_portable/store.py:432` |
| WebUI layout | ⚠️ 5 commits today (`585f382`..`b988b2d`) chase a data-prep page height clip (file library 989px not rendering); last observation: page still 437px tall after CSS changes — root cause not found. Header/tabs fixes (`a32c91a`..`c1f8872`) still need a visual re-check on fan-dragon |
| Models page | ✅ `/models` with category pills; trained exports resolved via project lookup (`aec11e6`..`25f3a3e`) |
| Data-prep chat | ✅ Implemented in `src/finetune_studio/webui/routes/data_prep_chat.py`; local model or external OpenAI-compatible API |
| Unload VRAM | ⚠️ Fixed in `a34c08b`; not regression-tested since 2026-09-10 |
| E2E suite (PHASES.md) | ⚠️ Stopped at Phase 3 (upload 10 fixtures to project `2026-09-10-oftest`, id `b08426e3`); Phases 3–8 pending |
| Deploy | ⚠️ fan-dragon runs a **bare** `uvicorn` (pid 926697, started 11:52, not systemd — `finetune-studio.service` does not exist there). `RESTART.md`/AGENTS deploy line saying `systemctl restart finetune-studio` is wrong |
| Docs | ✅ `AGENTS.md` is the source for commands/conventions; old handoff in `docs/archive/HANDOFF-2026-09-14.md` |

## Next steps
1. Fix the 17 F821 undefined names (list in State) — add the missing imports, nothing else — done when `.venv/bin/python -m ruff check src/ --select F821` prints `All checks passed!` and `.venv/bin/python -m pytest tests/test_vram.py -q -p no:cacheprovider` is all green (genorbox1)
2. Find the real cause of the data-prep page height clip — reproduce on fan-dragon with a Playwright `page.evaluate` that walks ancestors of the file library and prints each `clientHeight`/`overflow`/`display`; fix the one constraining element; stop the CSS-guess commits — done when the file library renders fully at 1280×800 and 1920×1080 (screenshots viewed) and the ancestor dump shows no clipped box
3. Doc fix: deploy is a bare uvicorn on fan-dragon (no systemd unit) — update `RESTART.md` + `AGENTS.md` Commands with the real restart (`pkill -f 'uvicorn finetune_studio' ; nohup .venv/bin/python -m uvicorn finetune_studio.webui.app:app --host 0.0.0.0 --port 7860 --no-access-log &`) or create the unit via `install-service.sh` — done when the documented command actually restarts the app and `ss -ltnp | grep 7860` shows the new pid
4. Ruff auto-fix pass — `.venv/bin/python -m ruff check src/ --fix` (F401/I001/UP045/RUF100/F541 only, review diff) — done when findings drop from 622 to the BLE001/S110 class only and `pytest tests/ --ignore=tests/unit/test_vram_profiler.py --ignore=tests/test_breakpoints.py --ignore=tests/test_breakpoints_visual.py --ignore=tests/test_phase_bd.py --ignore=tests/test_phase_bd_api.py` still passes
5. Visual re-check of header fixes on fan-dragon — open `http://fan-dragon:7860/`, confirm all 9 project tabs scroll on one line, no header overlap, model pill only says ONLINE when a model is loaded — done when all three hold at 1280px and 1920px widths
6. Resume E2E Phase 3 — `tests/run_qa.sh` (rules in `tests/README_E2E.md`); upload the 10 `OCTOPUS-7741` fixtures from `tests/fixtures/` to project `2026-09-10-oftest`, run the Qwen Q&A pipeline, export JSONL — done when the JSONL exists and every row contains `OCTOPUS-7741`
7. E2E Phase 4 — start training on the exported JSONL with a small base model and low epochs — done when the loaded Qwen model is unloaded before training starts and the run reaches 100% in the Training tab
## Commands
- Test (genorbox1, pure): `.venv/bin/python -m pytest tests/ -q --tb=no -p no:cacheprovider --ignore=tests/unit/test_vram_profiler.py --ignore=tests/test_breakpoints.py --ignore=tests/test_breakpoints_visual.py --ignore=tests/test_phase_bd.py --ignore=tests/test_phase_bd_api.py`
- Test (fan-dragon, GPU + browser): `ssh fan-dragon 'cd /home/genortg/finetune-studio && .venv/bin/python -m pytest tests/ -q --tb=short -p no:cacheprovider'`
- Deploy: `git push`, then `ssh fan-dragon 'cd /home/genortg/finetune-studio && git pull'` and restart the bare uvicorn (see Next step 3 — no systemd unit exists). Never `make run` on genorbox1 (dev-only box, no models/GPU)
- Verify (genorbox1): `.venv/bin/python -m ruff check src/ --select F821 && <Test (genorbox1) line>`; visual/E2E only on fan-dragon (`http://fan-dragon:7860/`)

## Blockers
- genorbox1 is dev-only (no install, no GPU, no models): pure unit tests + ruff here; GPU tests, Playwright tests, E2E Phases 3–8 and every visual check on fan-dragon after `git pull`
