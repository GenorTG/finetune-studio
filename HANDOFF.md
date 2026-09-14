# HANDOFF — finetune-studio

## Mission
Ships a local fine-tune + data-prep WebUI (FastAPI, `src/finetune_studio/`) with a data-prep tool-calling chat, training/merge/GGUF export, and an E2E browser test suite.
Genor edits on genorbox1 and pushes to GitHub; fan-dragon pulls and runs it as `finetune-studio.service` on port 7860 (RTX 3090).

## State (verified 2026-09-14)
| Area | Status |
|------|--------|
| Git | `main` @ `a32c91a`, clean, in sync with `origin/main` (`app.css` was reported dirty at handoff time but `git status` showed no local change — re-check before staging) |
| Unit tests | ❌ Not runnable on genorbox1: `.venv` lacks pytest (`No module named pytest`). Last known (2026-09-10): 167 pass / 37 fail, failures in `tests/unit/test_vram_profiler.py` (GPU-only) |
| WebUI header/tabs | ✅ Last 5 commits fixed tab-strip wrap/overflow, header overlap, false "MODEL ONLINE" pill (`a32c91a`..`c1f8872`) — needs a visual re-check on fan-dragon |
| Models page | ✅ `/models` with category pills; trained exports resolved via project lookup (`aec11e6`..`25f3a3e`) |
| Data-prep chat | ✅ Implemented in `src/finetune_studio/webui/routes/data_prep_chat.py`; local model or external OpenAI-compatible API |
| Unload VRAM | ⚠️ Fixed in `a34c08b`; not regression-tested since 2026-09-10 |
| E2E suite (PHASES.md) | ⚠️ Stopped at Phase 3 (upload 10 fixtures to project `2026-09-10-oftest`, id `b08426e3`); Phases 3–8 pending |
| Docs | ✅ `AGENTS.md` is the source for commands/conventions; old handoff in `docs/archive/HANDOFF-2026-09-14.md` |

## Next steps
1. Restore the test runner — `.venv/bin/python -m pip install -e ".[dev]"` then `.venv/bin/python -m pytest tests/ -q --tb=no -p no:cacheprovider` — done when pytest prints a pass/fail summary line; record it in the State table
2. Lint the tree — `.venv/bin/python -m ruff check src/` — done when it reports `All checks passed!`
3. Visual re-check of header fixes on fan-dragon — open `http://fan-dragon:7860/`, confirm all 9 project tabs scroll on one line, no header overlap, model pill only says ONLINE when a model is loaded — done when all three hold at 1280px and 1920px widths
4. Resume E2E Phase 3 — `tests/run_qa.sh` (rules in `tests/README_E2E.md`); upload the 10 `OCTOPUS-7741` fixtures from `tests/fixtures/` to project `2026-09-10-oftest`, run the Qwen Q&A pipeline, export JSONL — done when the JSONL exists and every row contains `OCTOPUS-7741`
5. E2E Phase 4 — start training on the exported JSONL with a small base model and low epochs — done when the loaded Qwen model is unloaded before training starts and the run reaches 100% in the Training tab
6. E2E Phases 5–7 — merge LoRA, export GGUF Q8_0, benchmark, load the export and ask 3–5 `OCTOPUS-7741` questions — done when the export is listed on `/models` and the answers contain training-only facts
7. Regression-test Unload — load any model in the WebUI, click Unload, check `nvidia-smi` on fan-dragon — done when VRAM used returns to the pre-load baseline

## Commands
- Test: `.venv/bin/python -m pytest tests/ -q --tb=no -p no:cacheprovider` (after step 1; `make test` for verbose)
- Run: `make run` (= `bash run.sh`); deploy: `git push`, then on fan-dragon `cd /home/genortg/finetune-studio && git pull && sudo systemctl restart finetune-studio` (see `RESTART.md`)
- Verify: `.venv/bin/python -m ruff check src/ && .venv/bin/python -m pytest tests/ -q --ignore=tests/unit/test_vram_profiler.py`

## Blockers
- pytest missing from `.venv` on genorbox1 — install dev extras (`pip install -e ".[dev]"`, Next step 1)
- GPU-dependent tests (`tests/unit/test_vram_profiler.py`) and E2E Phases 3–8 need fan-dragon — run there after `git pull`; if SSH is flaky, do code-only work locally
