# AGENTS.md — finetune-studio

Local fine-tune + data-prep + RAG WebUI (FastAPI + Jinja, Python 3.12, uv venv `.venv/`), CLI `fts`.
Hosts: edit/commit/push on **genorbox1** (`/home/genorbox1/work/finetune-studio`, RTX 3090 + GTX 1070);
**fan-dragon** (`/home/genortg/finetune-studio`) is deploy/test only — pull there, never SSH-edit files.
Both run the `finetune-studio` systemd **user** unit on :7860.

## Read first
1. `HANDOFF.md` — current state, in-flight work, next steps (≤120 lines; longer = stale, rewrite it).
2. `docs/WORKPLAN.md` — iron rules (binding) + ordered plan (last updated 2026-10-02; HANDOFF has newer state).
3. `docs/PRODUCT-BRIEF.md` — north star / quality bar. `docs/GOTCHAS.md` — full learned-rule log.
4. On demand: `docs/README.md` (doc map), `docs/CODEMAP.md` (symbol map), `RESTART.md` (service ops).

## Commands
- Setup/repair: `bash install.sh` (`--plan` preview GPU plan, `--check`, `--repair`, `--cpu`, `--gpu <vendor>`); dev extras: `uv pip install --python .venv/bin/python -e '.[dev]'`.
- Tests: `.venv/bin/python -m pytest -q -p no:cacheprovider --ignore=tests/test_vram.py` (full suite ~21 min); one file: `.venv/bin/python -m pytest tests/test_x.py -q`. `make test` = verbose full run incl. GPU tests.
- Lint: `.venv/bin/ruff check src/ scripts/` (must stay clean; `tests/` has ~108 legacy findings). `make lint` hides failures (`|| true`) — never trust it.
- Run: `make run` (= `bash run.sh`, uvicorn :7860). Prefer the service: `systemctl --user restart finetune-studio`; logs `journalctl --user -u finetune-studio -f`.
- Accelerator check: `.venv/bin/fts accel` (nonzero only on a GPU host that fell back to CPU).
- Symbols: `make codemap` / `.venv/bin/python scripts/codemap.py --grep NAME`; `make codemap-check` fails on drift.
- E2E browser suite: `tests/run_qa.sh` (see `tests/README_E2E.md`; can mutate live data).
- Deploy: `git push`, then on the host `bash update.sh` (non-interactive: pull ff-only, venv health, deps + torch pin, llama.cpp, DB migrate, restart). (Re)install unit: `bash install-service.sh`. Details: `RESTART.md`, `docs/DEPLOYMENT.md`.
- Verify before "done": the changed module's test file + ruff pass; GPU-path changes need a real GPU run noted in HANDOFF.

## Layout
- `src/finetune_studio/` — `webui/` (app, routes, templates, static), `cli/` (`fts` commands), `training/` (engine, worker, vram, export), `models/` (loaders, providers, registry), `data/` (parsers, prep, fs, `rag_portable/`), `rag/`, `testing/` (inference, judge), `benchmarks/`, `compare/`, `db/`, `accel/` (vendor-neutral GPU layer), `naming.py`, `config.py`.
- `tests/` — pytest, one file per concern; `e2e_*.py` + `run_qa.sh` are live/browser suites.
- `scripts/` — `accel_plan.py` (GPU detect/plan/install), `install_diagnose.py`, `codemap.py`, `git-hooks/`, dataset tools.
- `install*.{sh,ps1,bat,fish,zsh}`, `run*.{sh,bat,fish,zsh}`, `update.sh`, `install-service.sh` — keep shell variants in sync when changing one.
- Runtime (never committed): `data/`, `datasets/`, `models/`, `media/`, `output/`, `projects/`, `.llama.cpp/`, `.tmp/`; app state under `$FTS_ROOT` (default `~/.finetune-studio`).

## Conventions
- New code goes in the module that already owns the concern (check CODEMAP); new feature → new module + new test file, not a bigger file. Run `make codemap` after adding/moving/renaming symbols.
- Type hints on every signature; Pydantic/dataclass models for anything crossing the API boundary.
- HTTP: reuse what the module already uses (`urllib.request`/`httpx`); do not add a new HTTP library.
- Device code goes through `finetune_studio.accel` — never hard-code `cuda`, `{"": 0}`, `bf16=True` or bitsandbytes.
- Errors are honest: real 4xx/5xx with a reason, never 200 `{error}` or a silent fallback.
- Commits: imperative "what + why", one logical change each; the pre-commit hook bumps `VERSION` (run `make hooks` once per clone; skip with `FTS_NO_BUMP=1`).

## Session protocol
- Workspace `AGENTS.md` owns work modes / spawn rules. Here: goal + card, verify with Commands above.
- Cursor only if Genor asks: `ensure-cursor-helper.mjs --cwd /home/genorbox1/work/finetune-studio --label cursor-helper:finetune-studio`.
- End: rewrite `HANDOFF.md`, commit; push only with Genor's OK.
- HANDOFF sections, in order: `Mission` (2 lines), `State (verified <date>)` table, `In flight`, `Next steps` (≤7, exact commands), `Known issues`, `Commands`, `Blockers`. ≤120 lines; rewrite, never append; archive old copy to `docs/archive/HANDOFF-<date>.md`.

## Gotchas
<!-- One line each. Full context + history: docs/GOTCHAS.md — append there too. -->
- Dev docs are local-only: `docs/{ARCHITECTURE,CODEMAP,DEVELOPER,GOTCHAS,PRODUCT-BRIEF,README,REFACTOR-SPEC,WORKPLAN}.md` + `docs/{modules,audit,archive,judging}/` are gitignored — never `git add -f`, never link from README/Pages/INSTALL/DEPLOYMENT/TUTORIAL (`tests/test_repo_hygiene.py`).
- Never commit runtime data/corpora (`data/`, `datasets/`, `models/`, `media/`, `.venv`) — copy to fan-dragon by rsync; `.gitignore` re-includes need the three-step form (`data/benchmarks/default.json`).
- Install only via `install.sh`/`update.sh` (uv venv has no pip module; never `python -m pip`); every resolve passes `-c .venv/torch-constraints.txt` or torch silently drifts to another CUDA build.
- GPU stack choice lives only in `scripts/accel_plan.py` (tests forbid hard-coded `cuNNN`/`rocmX.Y` in installers); fake hardware with `FTS_ACCEL_FIXTURE=<json>`.
- uv: GPU llama-cpp wheel needs `--extra-index-url … --index-strategy first-index` (`--index-url` lets PyPI's CPU sdist win); source builds need `--no-cache`.
- Unsloth pins torch<2.13/transformers<=5.5: default `FTS_UNSLOTH=auto` skips it; `FTS_UNSLOTH=1` builds an older torch stack. torchaudio is intentionally not installed.
- genorbox1 GPU 1 is a GTX 1070: never use it — mask with `CUDA_VISIBLE_DEVICES`/`FTS_GPU_EXCLUDE="GTX 1070"` (the service unit does not mask it).
- Never kill foreign GPU/RAM users (games, ComfyUI, other services); no headroom → pause and report. Unload models/helpers you loaded.
- Sandbox instances: temp cwd + redirected `HOME`, `FTS_ROOT`, `FTS_DB` (main DB `data/finetune_studio.db` is cwd-relative); `FTS_ROOT` alone hits live data.
- Test screenshots go only to `.tmp/qa-shots/` (`FTS_QA_SHOTS`), never into docs/README/Pages/media; live E2E needs `FTS_ALLOW_LIVE_E2E=1`.
- Data-prep paths must go through `data.fs.paths.resolve_in_project`/`resolve_within`; no raw `Path(user_str)` in a route (`tests/test_data_prep_path_fence.py`).
- RAG export is AES-256-GCM encrypted by default; passphrase only in POST bodies, never stored/logged; shipped server binds 127.0.0.1 unless a token is set (`tests/test_rag_encrypted_package.py`).
- Studio corpus export is the encrypted `.ftsrag` bundle only (`rag_portable/secure_bundle.py`); never wire plaintext `store.export_bundle` to a route.
- Every project template declares `{% block workspace %}model|rag{% endblock %}` (RAG pages also `workspace_nav`); tab label, `<h1>` and `breadcrumb_tab` use the same word.
- Page headers: flow step + plain-language purpose + next-step link; jargon only in a `text-xs` line below.
- CSS: use only tokens defined in `app.css :root`/light block; keep the global `[hidden]{display:none!important}`; bump `app.css?v=` in `base.html` on every CSS change, and update every test pin of it in the same commit (`grep -rn '?v=' tests/`).
- Visual QA: `window.__audit` probe in dark AND light via the playwright-core runner in `~/.openclaw/workspace/.tmp/qa-sweep/`; on stubborn narrow-width bugs grep every `@media` block for the selector.
- Name artifacts with `naming.display_for_path()`/`resolve_run_path()` (never dir basenames, never `"/output" in path`); DB access only via `finetune_studio.db`.
- Match library files ↔ QA sources by `sha256` (manifests use `files/<sha12>/…`); RAG `document_id` is md5(path).
- Wizard chain: step 2 loads the helper, aborts if nothing was mined, unloads it before training — never let mining errors pass silently.
- fan-dragon login shell is fish: wrap multi-part SSH commands in `bash -c '…'`.
- Ruff `F821` here = a real `NameError` on an error path; fix it as a bug.
- OpenClaw tools: `exec` takes `timeoutSeconds`, not `timeout`; verify via `curl http://<host>:7860` first, browser second (fall back to curl after one timeout); don't narrate memory/runtime meta.
- `preset_advisor.py` step floor ignores `validation_split` and assumes eff. batch 8 (`steps = (pair_count * epochs) // eff_batch`): its `optimizer_steps` run ~2.2× high vs `metrics_json.total_steps` — treat as an upper bound until fixed.
- Models outside `models/` + `output/` → add a `model_dirs_extra` entry, `POST /api/models/refresh`, then reload the training page (its model dropdown is server-rendered at load).
- Training API start takes `model_path` (not `base_model`) + `preset_id` (e.g. `"qlora"`); GPU pinning goes in a `finetune-studio.service.d/*.conf` drop-in, never `systemctl --user set-environment`.
- RAG chat on a small tuned model ignores retrieved context when its training rows had none (Qwen3-0.6B plain rows: 2/5 grounded, with context-grounded rows 4/5 — `.tmp/rag-grounding/RESULTS.md`); the export's `grounded_share` option (`data/prep/grounding.py`, default 40% when a RAG corpus exists) fixes it at the data level. The rag/chat prompt layout lives only in `data/rag_portable/prompt.py` (pinned by `tests/test_rag_chat_prompt.py` + `test_grounded_rows.py`); moving CONTEXT into the user turn did not help — never rewrite it unmeasured.
