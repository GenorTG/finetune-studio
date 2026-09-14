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
- Every ~10 tool calls: `get_goal`, update the card. Same fix failed twice → stop and change approach.
- Heavy work (multi-file, long test loops): `sessions_spawn` with `cwd` = this repo; default model `opencode-go/glm-5.3-flash`.
- End: goal complete/blocked, card cleared, `HANDOFF.md` rewritten, commit + push.

## HANDOFF rules
- Sections, in order: `Mission` (2 lines), `State (verified <date>)` (table), `Next steps` (≤7, each with the exact command), `Commands`, `Blockers`. ≤120 lines. Rewrite, never append. Old content → `docs/archive/HANDOFF-<date>.md`.

## Gotchas
<!-- Append one line per learned rule. Format: "- <date> <rule> (<why/commit>)". -->
- 2026-09-10 `make lint` swallows failures with `|| true`; run ruff directly.
- 2026-09-10 37 tests in `test_vram_profiler.py` are GPU/env dependent — not a regression on genorbox1.
