# AGENTS.md — finetune-studio

Local fine-tune + data-prep WebUI (Python, `src/finetune_studio/`). Edit on **genorbox1**, commit + push, pull on **fan-dragon** for GPU runs (`finetune-studio.service`, port 7860). Never SSH into fan-dragon to edit files.

## Read first
1. `docs/WORKPLAN.md` — **the plan and its order is law** (durable across models; rules + numbered steps; never reorder).
2. `docs/PRODUCT-BRIEF.md` — what Genor wants the product to be good at (north star).
3. `HANDOFF.md` — verified ops state + next steps (≤120 lines; if longer, it is stale — rewrite it).
4. `docs/GOTCHAS.md` — full learned-rule log (AGENTS keeps only a short bootstrap subset).
5. `docs/README.md` — doc map. Then area docs as needed. `RESTART.md` = fan-dragon service. `PHASES.md` may be stale vs HANDOFF.

## Commands
- Tests: `make test` (= `.venv/bin/python -m pytest tests/ -v --tb=short`). Single file: `.venv/bin/python -m pytest tests/test_api.py -v`.
- Lint: `.venv/bin/python -m ruff check src/` (the Makefile `lint` target hides failures with `|| true` — run ruff directly and fix every warning).
- Run: `make run` (= `bash run.sh`). E2E browser suite: `tests/run_qa.sh` (see `tests/README_E2E.md`). GPU-dependent tests (`test_vram_profiler.py`) only pass on fan-dragon.
- Verify before "done": the test file for the module you changed passes locally; GPU-path changes need a fan-dragon run noted in HANDOFF.
- Deploy: `git push`, then on fan-dragon: `cd /home/genortg/finetune-studio && git pull --ff-only`. Restart with `systemctl --user restart finetune-studio` (user unit, no sudo; `journalctl --user -u finetune-studio -f` for logs). (Re)install the unit with `bash install-service.sh` on fan-dragon — it also stops a stray bare uvicorn on :7860. Truth check: `ss -ltnp | grep 7860` pid's `/proc/<pid>/cgroup` contains `finetune-studio.service`. Never `make run` on genorbox1 (dev-only box, no models/GPU).

## Layout
- `src/finetune_studio/` — app code, one concern per module (api, ui, training, data-prep). `tests/` mirrors it.
- `scripts/`, `install*.{sh,fish,zsh,ps1,bat}`, `run*.{sh,fish,zsh,bat}` — installers/launchers; keep all shell variants in sync when changing one.
- `data/`, `datasets/`, `models/`, `output/`, `projects/` — runtime artifacts, never committed.

## Working in this repo (all models)

- **`docs/CODEMAP.md` is the symbol map.** Before grepping around for "where is X / what
  imports Y", run `make codemap` (or `.venv/bin/python scripts/codemap.py --grep NAME`)
  — it indexes every module, class, function with signatures + intra-repo imports in one file.
  Grep mode: `scripts/codemap.py --grep score_results` → `file:line def name(sig)`.
- **Any session that adds/moves/renames modules, classes, or functions must run
  `make codemap` and commit the regenerated `docs/CODEMAP.md` in the same commit** as
  the code change. CI-style check exists: `make codemap-check` (fails if the committed
  map is stale).
- New code goes in the module that already owns that concern (one concern per module);
  CODEMAP shows who owns what at a glance. `src/finetune_studio/benchmarks/__init__.py`
  being 536 lines with everything in `__init__.py` is a known wart — do not add to it.

## Conventions
- Type hints on every function signature (params + return). Pydantic/dataclass models for anything crossing the API boundary.
- One module = one responsibility; new feature → new module + new test file, not a bigger existing file.
- Async I/O via the existing HTTP client wrapper; do not introduce a second HTTP library.
- Commits: imperative, "what + why", one logical change each.

## Session protocol
- Workspace `AGENTS.md` owns work modes / spawn truth / Cursor helpers. This repo: goal+card, then verify with Commands above.
- Cursor only if Genor asks: `ensure-cursor-helper.mjs --cwd /home/genorbox1/work/finetune-studio --label cursor-helper:finetune-studio` → `acpSessionKey`.
- End: goal done/blocked, card clear, rewrite `HANDOFF.md`, commit+push.

## HANDOFF rules
- Sections, in order: `Mission` (2 lines), `State (verified <date>)` (table), `Next steps` (≤7, each with the exact command), `Commands`, `Blockers`. ≤120 lines. Rewrite, never append. Old content → `docs/archive/HANDOFF-<date>.md`.
## Gotchas
<!-- Full log: docs/GOTCHAS.md — append there; keep ≤12 critical lines here for bootstrap. -->
- 2026-09-21 **Flow-scoped nav contract:** every project template MUST declare `{% block workspace %}model|rag{% endblock %}` — without it the flow subnav silently disappears (7 pages shipped that way) and the session strip falls back to the model flow. RAG-only pages also set `{% block workspace_nav %}rag{% endblock %}`. Tab label, page `<h1>`, and `breadcrumb_tab` must all use the SAME word (the `data`/`files`/`data-prep`/`pairs` mismatch is what made the app unreadable) (5615f73, d605f7e).
- 2026-09-21 **Page-header copy rule:** a project page header states (1) which numbered step of which flow it is, (2) what it does in plain words — no LoRA/corpus/"three layers" jargon in the first sentence, (3) a link to the next step. Jargon goes in a `text-xs` line below (ae44ac5).
- 2026-09-20 **CSS token discipline:** templates may only reference tokens defined in app.css `:root` / base.html light block — an undefined `var(--x, #fallback)` silently renders the dark-theme fallback in light mode (13 legacy names like `--accent-green`/`--bg-elev`/`--muted` shipped 1.3:1 text this way; now aliased to real tokens). Measure contrast against the COMPOSITED ancestor background, never the element's own `backgroundColor`.
- 2026-09-20 **Visual QA probe:** the `window.__audit` pattern (overflow past viewport + clipped cells + <10px text + WCAG vs composited bg), run on every route in dark AND light at 1270px and 320px, catches what pytest/HTTP codes never see (clipped action buttons, 300px silent truncations). `act kind=resize` does NOT change the viewport — use `browser action=emulate device="Desktop Chrome"|"iPhone SE"`.
- 2026-09-20 **hidden attr vs classes:** `.btn`/`.pill` set `display`, which beats the UA `[hidden]` rule — the global `[hidden]{display:none!important}` in app.css is load-bearing; never remove it, and bump `?v=` in base.html with every css change (stale cache hid this bug for a whole pass) (7e493c7).
- 2026-09-20 **Artifact naming:** never label models/datasets by directory basename — use `finetune_studio.naming.display_for_path()` (`<Project> · <Base> · [version] · [Kind] [QUANT] [abliterated]`); DB lookups must go through `finetune_studio.db`, never a hand-built sqlite path (the old lookup pointed at a nonexistent file and silently degraded every run-export name to a hash dir) (7777981).
- 2026-09-20 **Relative scan roots:** the deployed service walks `output/projects/…` with NO leading slash — never gate a path category on `"/output" in path`; lead with `naming.resolve_run_path()` (2b107ff; the first naming release still shipped hash labels live because of this).
- 2026-09-20 **Workbench matching rules:** QA source manifests point at content-addressed `files/<sha12>/…`, NOT the raw library path — match file↔source by `sha256`; and RAG corpus `document_id` is md5(path) — check corpus membership via manifest `documents_meta[].source` sha12 dirs (e7e9f7f, 266ffc1; both caught live during browser verification, not by tests).
- 2026-09-20 **Per-commit versioning:** run `make hooks` once per clone; the **pre-commit** hook bumps `VERSION` (BUILD segment) and the commit *includes* the bump — commit-msg cannot (git snapshots the index before it; the first versioning scheme shipped one build behind, caught by the 0.1.0.2 E2E). UI chip + `GET /api/system/version` show it. Skip a bump with `FTS_NO_BUMP=1 git commit …`; VERSION-only commits don't bump.
- 2026-09-20 **Wizard chain:** step 2 auto-loads the helper provider, aborts when no document could be mined, and unloads the helper before training. Never let mining errors pass silently — the E2E on fb05a690 caught the chain coasting into a coverage-fill-only dataset (87a8643).
- 2026-09-18 **Fan-dragon resources:** high RAM/VRAM often = **other services or games**. Never kill/interrupt those. If Finetune Studio cannot run for lack of headroom → **pause** and report. Inside the studio, load/unload models/helpers yourself (do not leave stacked loads).
- 2026-09-18 **MiniMax / OpenClaw tools:** never pass `timeout` to `exec` — use `timeoutSeconds` (integer). Prefer native tools over the `tool_call` meta-tool. Prefer `curl`/API on `http://fan-dragon:7860` over browser for verification; if browser times out once, fall back to curl immediately (do not retry CDP evaluate loops). Never narrate runtime/memory meta ("Per AGENTS.md…", "untrusted memories…", "context confirms clean…") — silent continue; call tools; answer Genor.
