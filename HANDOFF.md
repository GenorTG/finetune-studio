# HANDOFF — finetune-studio

Local fine-tune + data-prep WebUI (`src/finetune_studio/`).
Edit on **genorbox1** → push → **fan-dragon** runs GPU work; **both hosts run the service** via the repo scripts.

**Read first:** `docs/WORKPLAN.md` (order is law) · `docs/PRODUCT-BRIEF.md` (north star) · this file · `AGENTS.md`.

## Mission

Trained models must reliably answer the learned corpus — no lying about trained DB sources. Judge by reading transcripts, not auto-greens. **Data guarantee: every parsed chunk must reach the training dataset (no silent holes).**

## State (verified 2026-09-26 23:47 CEST · HEAD `b119b50` pushed · genorbox1 + fan-dragon both `active` on :7860, both probed 0/59 leaks)

| Area | Status |
|------|--------|
| **Project-scoped 404 UX is now complete** | `e37dd3c` closed the class `bc8d018` opened. Measured, not inferred: probe of all 59 project-scoped API GET routes with a bogus pid → **0 return 200** (was 24). Pages deliberately unchanged: 14 of 16 302 a bad pid to `/projects` by design, 2 return 404. Guard mirrors `file_library.py:48` `_project_or_404`, each file keeping its local error style. `/api/projects/{pid}/rag` no longer hands an absolute `corpus_dir` path for a nonexistent project. The 3 SSE routes guard in the **outer** function, so no `200 + data:` error frame is ever emitted. A guard only changes behaviour for a pid that does not exist, so valid-project flows are untouched (a real project with no runs still returns `[]`). |
| **Both hosts deployed + verified identically** | fan-dragon `update.sh` green 23:45, HEAD `b119b50`, service `active`; VRAM observed 4.6 G/16 G before touching anything (never killed a foreign user). genorbox1 needed a restart — it had been serving code from 19:04, i.e. *before* the fix landed. |
| **`make test` no longer eats its own failure** | `b119b50`: the target was `... pytest ... \|\| python -m pytest ...`. The fallback reaches for a bare `python`, absent from this venv, so a genuinely failing suite exited **127** and the pytest exit code was discarded — it only ever bit on the runs that mattered. Real code now, so a red suite surfaces its own failure. |
| **⚠️ The CPU suite is NOT green — 9 pre-existing failures** | Older revisions of this file claimed "1083 pass". It is not true and has not been for a while. Full run at `b119b50`: **9 failed, 1138 passed**. All 9 are pre-existing and unrelated to the 404 work (that diff touches no template or asset file). 7 are stale-test drift, e.g. `base.html` ships `app.css?v=66` while `test_ui_reliability::test_css_cache_bust_bumped` and `test_header_nav::test_css_cache_bust_header_nav` still assert `?v=65` — drift introduced in `8d2efae` (2026-09-25). Also `test_bench_judge::test_bench_judge_records_verdict_in_db`, `test_export::TestExportHelpers::test_find_tool_returns_none_when_absent`, `test_install_diagnose` ×2, `test_header_nav::test_header_nav_overflow_affordance_markup`. The other 2 (`test_merge_base_resolution`, `test_sft_args`) are **order-dependent**: green in isolation, red in a full run. Untriaged — see Next steps. |
| **Lint baseline** | `ruff check src/` = **134 pre-existing findings**, unchanged by `e37dd3c`/`b119b50`. Do not blind `--fix` (that is how the 98 F401s in `db/__init__.py` nearly deleted the DB facade). |
| **Hardware reality** | genorbox1: **RTX 3090 24 GB**, driver 535 → cu124 torch ceiling 2.6.0, **1× M.2 occupied (228 GB root @ 85 %)** + 1× U.2. fan-dragon: **RTX 5080 16 GB**, driver 615.71. GPU tests (`test_vram_profiler.py`) only pass on fan-dragon. |
| **Prior verified pillars** | Coverage-fill 100% gate (`6af6312`+`451facd`); RAG standalone MCP pkg (`90bcc21`,`2b5235d`,`784efe8`); project versions + wizard (`bc1152b`,`90c86dc`); size-aware preset advisor (`b78bf15`); flow-scoped nav + copy (`291b1a4`→`d605f7e`); artifact naming (`7777981`); file workbench (`8efaa80`→`266ffc1`); per-commit VERSION bump via pre-commit (`make hooks`); import-health guard (`9903891`); DB isolation autouse (`d730456`,`be776ff`); hand-rolled client fixture fix (`26199a3`,`a59fe69`). Training: `f76bf64f` (12 ep r128) = 94.8 % on the locked 500-case instrument. |

## Next steps

1. **Triage the 9 red CPU tests** (do this before anything claims the suite is green). Start with the 5 trivially stale cache-bust/affordance asserts, then `test_bench_judge` and `test_install_diagnose`; the 2 order-dependent ones are a test-isolation bug in the same family as the `d730456` DB-isolation work.
2. **Versions UX polish** — compare manifests side-by-side, copy-pins-to-new-project: `webui/routes/versions.py` + wizard step 6.
3. **Specialized RAG corpus** for the 10 hand-picked 58d4e331 files: `POST /api/projects/58d4e331/rag/build` on a filtered source set; pin in a version manifest.
4. **60ep/r64 noise-check** (GPU, fan-dragon): `POST /api/training/start {project_id:"58d4e331", dataset_id:<a0ae8778>, preset_id:"standard", overrides:{num_epochs:60, lora_rank:64}}`.
5. **Eyeball bench judging** on the NEW 554-case full-coverage instrument (`f76bf64f/merged`) per `docs/judging/PROTOCOL.md` — first comparable full-coverage verdict (GPU).
6. **genorbox1 disk**: ~30 G free — no model downloads until Genor's SSD decision lands.

## Commands

```bash
# both hosts — install/update/service (non-interactive)
bash install.sh && bash install.sh --check     # deep health (torch pin, llama-cpp, CLI)
bash update.sh                                  # pull + health + pinned sync + migrations + restart
systemctl --user restart finetune-studio       # genorbox1 AND fan-dragon (user units)

# dev loop (genorbox1)
make test                  # CPU suite — real exit code now; expect the 9 known reds
.venv/bin/python -m ruff check src/   # NEVER `make lint` (it ends in `|| true`)
make codemap / make hooks

# the 404 audit (dev tool, untracked, not in git)
.venv/bin/python .tmp/audit/probe_pid_routes.py http://127.0.0.1:7860
# GET-only; walks the app's own route table. FastAPI 0.141 nests routers as
# _IncludedRouter, so a flat `app.routes` scan sees only the 5 built-ins —
# real paths live on `original_router.routes` under `include_context.prefix`.
# Expect: 0/59 API leaks, page_status_counts {'302': 14, '404': 2}.

# deploy
git push && ssh fan-dragon 'cd /home/genortg/finetune-studio && bash update.sh 2>&1 | tail -5; git log --oneline -1'
```

## Blockers

- **fan-dragon VRAM** — free enough to deploy (4.6 G/16 G used), but model pulls stay parked until Genor says so. Rule stands: observe, never kill a foreign user.
- **genorbox1 disk 85 %** → no model pulls; NVMe expansion is Genor's hardware call (PCIEX4 M.2 adapter is the cheap path).
- driver 535 caps torch at cu124 (2.6.0); newer torch needs driver ≥ 580 — only if a feature demands it.
- Visual changes still require real rendered-page checks (sweep + `shots/*.png`); HTTP codes alone don't count.

## Gotchas worth re-reading before data-prep or training work

See repo `AGENTS.md ## Gotchas` + `docs/GOTCHAS.md`. Key: coverage-fill counts `qa` and `qa_coverage_fill` separately; extractive fill answers stay verbatim; never compare strict % across different suite instruments; templates may only use defined CSS tokens; artifact labels via `naming.display_for_path()`. **`benchmarks.py` routes returning `JSONResponse` need `response_model=None`** — a `-> list[...] | JSONResponse` annotation is not a valid Pydantic field and the app fails to *import* without it.
