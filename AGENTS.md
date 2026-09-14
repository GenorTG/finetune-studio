# AGENTS.md — finetune-studio

Local fine-tune + data-prep WebUI (Python, `src/finetune_studio/`). Edit on **genorbox1**, commit + push, pull on **fan-dragon** for GPU runs (`finetune-studio.service`, port 7860). Never SSH into fan-dragon to edit files.

## Read first
1. `HANDOFF.md` — current state and next steps (≤120 lines; if longer, it is stale — rewrite it).
2. `docs/` for the area you touch. `PHASES.md` = roadmap, `RESTART.md` = service restart on fan-dragon.

## Commands
- Tests: `make test` (= `.venv/bin/python -m pytest tests/ -v --tb=short`). Single file: `.venv/bin/python -m pytest tests/test_api.py -v`.
- Lint: `.venv/bin/python -m ruff check src/` (the Makefile `lint` target hides failures with `|| true` — run ruff directly and fix every warning).
- Run: `make run` (= `bash run.sh`). E2E browser suite: `tests/run_qa.sh` (see `tests/README_E2E.md`). GPU-dependent tests (`test_vram_profiler.py`) only pass on fan-dragon.
- Verify before "done": the test file for the module you changed passes locally; GPU-path changes need a fan-dragon run noted in HANDOFF.
- Deploy: `git push`, then on fan-dragon: `cd /home/genortg/finetune-studio && git pull --ff-only`. Restart the bare uvicorn per `RESTART.md` "Start / restart the WebUI on :7860" (start new → kill old pid → `ss -ltnp | grep 7860` proves new pid). **fan-dragon has no `finetune-studio.service` by default**; to create one, `bash scripts/install-service.sh` on fan-dragon (requires sudo). Never `make run` on genorbox1 (dev-only box, no models/GPU).

## Layout
- `src/finetune_studio/` — app code, one concern per module (api, ui, training, data-prep). `tests/` mirrors it.
- `scripts/`, `install*.{sh,fish,zsh,ps1,bat}`, `run*.{sh,fish,zsh,bat}` — installers/launchers; keep all shell variants in sync when changing one.
- `data/`, `datasets/`, `models/`, `output/`, `projects/` — runtime artifacts, never committed.

## Conventions
- Type hints on every function signature (params + return). Pydantic/dataclass models for anything crossing the API boundary.
- One module = one responsibility; new feature → new module + new test file, not a bigger existing file.
- Async I/O via the existing HTTP client wrapper; do not introduce a second HTTP library.
- Commits: imperative, "what + why", one logical change each.

## Session protocol
- Start: `create_goal` (objective + acceptance command), `progress_card` ≤7 steps. Clear any stale card first.
- Every new instruction from Genor = new task: `get_goal` first; if objective mismatches, ask Genor for `/goal edit …` (model cannot rewrite objective — only complete/blocked) and reset the `progress_card` before any other call. Every ~10 tool calls: `get_goal`, update the card. Same fix failed twice → stop and change approach.
- Implementation is Cursor's, mechanically: hand-edit only one-file fixes ≤30 lines, docs, config. New feature / new file, scene or test / ≥2 code files / a fix that already failed once → `sessions_spawn` with `runtime: "acp"`, `agentId: "cursor"`, `streamTo: "parent"`, `label`, `cwd` = this repo (Cursor Auto, flat subscription); then `sessions_yield`. You reproduce, review the diff, verify, commit, push. Cheap research/triage only: `runtime: "subagent"`, `model: "opencode-go/deepseek-v4-flash"`.
- End: goal complete/blocked, card cleared, `HANDOFF.md` rewritten, commit + push.

## HANDOFF rules
- Sections, in order: `Mission` (2 lines), `State (verified <date>)` (table), `Next steps` (≤7, each with the exact command), `Commands`, `Blockers`. ≤120 lines. Rewrite, never append. Old content → `docs/archive/HANDOFF-<date>.md`.

## Gotchas
<!-- Append one line per learned rule. Format: "- <date> <rule> (<why/commit>)". -->
- 2026-09-10 `make lint` swallows failures with `|| true`; run ruff directly.
- 2026-09-10 37 tests in `test_vram_profiler.py` are GPU/env dependent — not a regression on genorbox1.
- 2026-09-14 genorbox1 is dev-only: pure pytest + ruff here (`uv pip install --python .venv/bin/python -e .[dev]` — the venv is uv-managed, `python -m pip` does not exist); anything needing GPU, Playwright or the running app → fan-dragon.
- 2026-09-14 fan-dragon has no `finetune-studio.service`; the app is a bare `uvicorn` process — check `ss -ltnp | grep 7860` before claiming a restart worked.
- 2026-09-14 Ruff `F821` (undefined name) in this repo = real `NameError` on error paths (missing imports); treat as bugs, not style.
- 2026-09-14 fan-dragon restart: `pkill -f "uvicorn finetune_studio"` *before* spawning the new process will match the child's argv and race-kill the parent bash (exit 255). Start the new process first, wait 3s for it to bind, then `kill $OLD_PID` explicitly. Truth check is `ss -ltnp | grep 7860` — new pid must differ from old.
- 2026-09-14 Data-prep file library was clipped by `<main class="content">` (`flex:1 1 0` + `overflow:hidden auto` in a 361px grid row). Fix landed in `4bebc37`; ancestor walk via browser `page.evaluate` is the diagnostic pattern for any future "page stuck at 437px" symptom — not more CSS guesses.
- 2026-09-14 `db/__init__.py` only re-exports a curated symbol subset. Some symbols live only on `finetune_studio.db.datasets` etc. — call sites must use the bare imported name, not `db.get_X(...)`. Symptom in tests = `AttributeError: module 'finetune_studio.db' has no attribute 'get_dataset_by_path'`. QABUG-001 fix precedent: import `get_dataset_by_path` from `finetune_studio.db.datasets` and call it bare.
- 2026-09-14 `pfs.list_qa_sources(pid)` (data-prep source-picker) is a different code path from `fl.list_files(pid)` (file library). Newly uploaded files appear in the file library but NOT in the data-prep source-picker until they go through the prep-parser pipeline. Use `POST /api/projects/{pid}/data-prep/sources` to promote a `file_id` into a parsed-source row (QABUG-003 fix).
- 2026-09-14 Top-nav `[tools] hf inference` link points to `/projects/{pid}/inference` which is 404 — the real chat-against-model surface is `/projects/{pid}/chat`. Remap the nav link (QABUG-004) or add a real inference page.
- 2026-09-14 FastAPI multipart upload field is `files=` (plural). `file=` singular gets 422 from FastAPI. The `routes/file_library.py:1-25` docstring lists it correctly; legacy scripts that guess wrong get a clear 422.
- 2026-09-14 Browser snapshot tool schema rejects `elementLine` arg as `id`; the schema requires the `id` field to be set. Use the `depth` arg alone, or fall back to `curl + python3 strip` for bodyText probes when the browser path fails.
- 2026-09-14 Browser `emulate` rejects `width`/`height` bare integers (`must be integer`); `window.resizeTo(1920,1080)` is blocked by Chromium non-user scripts. Evidence ceiling for single-resolution captures = 1 screenshot per page + 1 bodyText dump.
