# Handoff — finetune-studio

## Mission

Local fine-tune, data-prep and RAG WebUI. GPU-first; never silently fall back.
Commits, pushes, local service restarts and real-hardware tests are authorized.

## State (branch `feat/app-guide`, verified 2026-10-06; `main` = `27c881f`, live service runs `87497e6`)

| Area | State |
|---|---|
| Guide (new) | Agent mode is now an app-wide **Guide**: docked panel on every page (`? guide` button), `finetune_studio/guide/` (KB, registry, tools, loop), `POST /api/guide/chat` (SSE), `static/js/guide.js`. Not merged, not deployed. |
| KB | 19 markdown entries in `guide/kb/` (one per page/feature + dataset quality, training settings/routes, helper, troubleshooting); BM25 `app_help` (no embedder/GPU). Every `#id`/`[name]` listed is test-verified against the real template. |
| Tools | Read-only: `app_help`, `get_app_guide`, `project_overview`, `inspect_project_readiness`, `list_datasets`, `list_runs`, `system_status`, `recommend_training`, `dataset_health`, `explain_setting`, `list_sources`, `read_source`, `list_qa_pairs`. UI (allow-listed): `navigate`, `highlight`, `suggest_settings` (pre-fill only). Only mutation: `create_qa_pairs` (pending rows). |
| Behaviour | Forced final-answer turn at the round limit (live 6-round no-reply bug); dead helper → error, not empty reply; readiness reply still the server summary; helper-model rule unchanged (explicit helper, 409 when absent). |
| Tests | `tests/test_guide_{kb,registry,state_tools,ui_tools,loop,templates}.py` + updated `test_agent_*`; ruff on `src/ scripts/` clean. Sandbox browser E2E `tests/e2e_guide_sandbox.py` 40/40 (dark+light, `__audit` clean). |
| Service | genorbox1 :7860 untouched (still `87497e6`); GTX 1070 foreign workload left alone. |

## Design decisions

- **Persistence:** the panel lives in the base shell outside `#content`; spa.js only swaps `#content`, so in-app navigation never unloads it (no restore step). `sessionStorage` mirrors history, open state and one pending UI effect for hard reloads / SPA full-load fallbacks (tab-scoped, no stale days-old chats).
- **Agent radio on Chat** opens the same panel (one conversation, one renderer); the chat page's own agent request path and tool renderer were removed.
- **Safety:** the model can only name registered pages/controls/fields; values are range/choice checked; `guide.js` never clicks or submits; there is no start/approve/export/delete tool (pinned by test).
- **Streaming:** SSE events `start/status/thinking/tool_call/tool_result/ui/final/error/done`; a slow helper load happens inside the stream (instant "loading helper" feedback). Model tokens are not streamed yet (one blocking model call per round).

## Next steps

1. Genor: live answer-quality evaluation with Gemma-12B (questions in `tests/E2E_MANUAL_GUIDE.md` §12); tune `guide/prompt.py` and KB wording from what it gets wrong.
2. Decide merge of `feat/app-guide` → `main`, then `systemctl --user restart finetune-studio` (Genor's OK) and re-run `tests/e2e_guide_sandbox.py`.
3. Optional: stream model tokens; guided "walk me through" multi-step mode; mutating tools only with an explicit confirm step.
4. Fix spa.js re-entry of full-loaded pages (column-0 `const` → full reload, cuts an in-flight stream; see AGENTS Gotchas).
5. Official GSM8K full run remains unverified (bounded samples first); `ci-ok` not required by `main` protection.
6. fan-dragon deploy stays deferred unless Genor asks.

## Commands

- Guide tests: `PYTHONPATH=$PWD/src .venv/bin/python -m pytest -q -p no:cacheprovider tests/test_guide_*.py tests/test_agent_*.py tests/test_data_prep_chat*.py`; lint `.venv/bin/ruff check src/ scripts/`.
- Sandbox browser proof: `PYTHONPATH=src .venv/bin/python tests/e2e_guide_sandbox.py --port 7896` (fake helper, temp cwd/HOME/FTS_ROOT/FTS_DB; shots in `.tmp/qa-shots/guide/`).
- Manual walkthrough and route standards: `tests/E2E_MANUAL_GUIDE.md`; live browser phases: `FTS_ALLOW_LIVE_E2E=1 .venv/bin/python tests/e2e_user_walkthrough.py --list`.
- Install: `bash install.sh --verify`; `.venv/bin/python scripts/install_diagnose.py --check --no-service-check`.

## Known issues

- The advisor's `optimizer_steps` is an upper bound (ignores the validation split); `recommend_training` says so.
- Guide answers depend on the helper model following the `<tool_call>` format; a weak model can loop (duplicate-call guard + forced final turn bound the damage).
- Preference-comparison authoring, ORPO/KTO, automatic teacher traces and eval decontamination are not implemented (the KB says so).
- DPO smoke logged TRL Qwen tokenizer prefix-mismatch warnings; 16-row run proves plumbing only.
