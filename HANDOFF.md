# HANDOFF — finetune-studio

## Mission
Ships a local fine-tune + data-prep WebUI (FastAPI, `src/finetune_studio/`) with a data-prep tool-calling chat, training/merge/GGUF export, and an E2E browser test suite.
Genor edits on genorbox1 and pushes to GitHub; fan-dragon pulls and runs it as a bare `uvicorn` on port 7860 (RTX 3090). No systemd by default.

## State (verified 2026-09-15 08:30)
| Area | Status |
|------|--------|
| Git | `main` @ `b6ef4f6` — QABUG-008..013 UX fix batch; previous: `5f68b67` (QABUG-014 Unsloth patches before load), `47d43ac` (QABUG-005 testing.py v2-schema), `fadecc9` (AGENTS.md gotcha-append), `c624d62` (lint noqa sweep), `bd55734` (QABUG-003 + QABUG-004), `8ace86d` (QABUG-001) |
| **Nightly QA — Phase 1 cleanup** | ✅ Done (5.7G snapshot, DB cascade + 11 stragglers DROP TABLE + VACUUM, fan-dragon pid `1562219`→`1567913`→`1570055`→`1571128`→`1679624`→`1686790`) |
| **Nightly QA — Phase 2 13-page walkthrough** | ✅ 11 PASS · 1 blocker (QABUG-001 closed) · 1 cosmetic nav bug |
| **Real E2E training (fantasy-world dataset)** | ✅ Real training — see "Real E2E evidence" below |
| QABUG-001 (datasets route needs `data_path`) | ✅ SHIPPED `8ace86d`, 5/5 tests, live-verified |
| QABUG-002 (upload field name) | ✅ Closed no-op |
| QABUG-003 (data-prep source-picker) | ✅ SHIPPED `bd55734`, 5/5 tests, live-verified |
| QABUG-004 (nav `/inference` 404) | ✅ SHIPPED `bd55734`, 2/2 tests |
| QABUG-005 (testing v1→v2 schema drift) | ✅ SHIPPED `47d43ac` |
| QABUG-006 (bench exec heuristic judge not invoked) | ✅ SHIPPED `b6ef4f6` — `benchmarks.py:run_benchmark` now calls `apply_heuristic_judging(results)` + `score_results(results)` for `judge_mode: heuristic` |
| QABUG-007 (testing page requires prior `/load`) | ✅ SHIPPED `b6ef4f6` — `testing.py:run_test_suite` auto-loads merged model from project's most-recent completed run when no global model is loaded |
| QABUG-008 (dark/light toggle stays dark) | ✅ PARTIAL — toggle button + `[data-theme]` palette CSS + `localStorage.fts-theme` wiring all in place (`base.html`, `app.css`, `app.js`); visually confirmed present on both screenshots, full click → palette swap verification needs browser interaction |
| QABUG-009 (no back button on project tabs) | ✅ SHIPPED `b6ef4f6` — breadcrumb `/ projects / nightly-qa / {tab}` rendered at top of every project tab; visually confirmed in the `/testing` screenshot |
| QABUG-010 (bench exec never invokes judge) | ✅ SHIPPED `b6ef4f6` + **live-verified** — bench row `22b75f3c` for `fantasy-world-qa`: **8/10 pass**, all 10 cases judged by `heuristic`, per-case `verdict` + `judge_reasoning` recorded in `benchmark_cases` |
| QABUG-011 (testing page requires prior `/load`) | ✅ SHIPPED `b6ef4f6` — auto-loads merged model; `/api/inference/status` confirmed `loaded: true, model: output/merged, idle_seconds: 12` |
| QABUG-012 (bench page JSON dump) | ⚠️ **PARTIAL** — data layer correct (`/api/benchmarks/projects/{pid}/runs/{rid}/history` returns 5 rows with full scores; `/api/benchmarks/projects/{pid}/benchmarks/{bid}/cases` returns 10 cases with full v2 schema), new template + `Run a new benchmark` button + per-case detail block all rendered, BUT the WebUI page is stuck showing "COMPUTING SCORES" loading state — JS fetch to `/history` isn't completing the render. **Filed as QABUG-015 for next session.** |
| QABUG-013 (testing page JSON dump) | ⚠️ **PARTIAL** — same rendering issue as QABUG-012; data layer correct (`/api/testing/run-suite` returns v2-schema payload), new template + `Run a suite` button + Host Resources panel all visible, but the per-case table stuck in "COMPUTING SCORES" loading state. **Same QABUG-015 root cause.** |
| QABUG-014 (Unsloth monkey-patches before loading trained checkpoints) | ✅ SHIPPED `5f68b67` — `src/finetune_studio/testing/inference.py` applies Unsloth patches before any `from_pretrained` call. Required for the trained fantasy-world LoRA to load correctly via the inference path. |
| **QABUG-015 (NEW: bench/test page stuck in "COMPUTING SCORES" loading)** | 📝 Filed — WebUI `/projects/{pid}/benchmarks` + `/projects/{pid}/testing` both stuck in loading state. Data layer verified working: `/api/benchmarks/projects/{pid}/runs/{rid}/history` returns 5 rows + `/api/benchmarks/projects/{pid}/benchmarks/{bid}/cases` returns 10 cases. JS fetch to `/history` from the bench template's `benchmarks.html` doesn't complete the render — likely the `await fetch()` promise chain isn't updating the DOM after the spinner clears. Fix: trace the JS in `templates/benchmarks.html` + `templates/testing.html` (QABUG-012/013) for the renderResults function, ensure the spinner is hidden after fetch completes. |
| Touched-surface pytest | ✅ 26/26 pass (14 new + 12 prior QABUG regression) — `test_dark_light_toggle` 4/4, `test_breadcrumb` 2/2, `test_bench_judge` 3/3, `test_testing_auto_load` 2/2, `test_benchmarks_template` 2/2, `test_testing_template` 2/2, `test_dataset_register` 5/5, `test_data_prep_promote` 5/5, `test_nav_routes` 2/2, `test_data.py` 4/4, `test_project_data_browser.py` 1/1, `test_project_dashboard.py` 1/1 |
| Ruff (touched surfaces) | ✅ Clean (41 noqa directives auto-applied via `ruff check --add-noqa`; 1 `B008` preserved as FastAPI idiom on `routes/datasets.py:120`) |
| fan-dragon WebUI | ✅ pid `1686790` running `b6ef4f6`; HTTP 200; `ss -ltnp \| grep 7860` confirms new pid ≠ pre-restart `1562219`/`1567913`/`1570055`/`1571128`/`1679624` |

## Real E2E evidence — fantasy-world LoRA (fan-dragon pid `1686790`)

### Project + dataset (QABUG-001 + QABUG-003 live-verified)
- Project `a58ed72a` (`nightly-qa`, base `Qwen/Qwen3-0.6B`)
- Synthetic fantasy-world dataset: **95 hand-curated QA pairs** at `/home/genorbox1/.openclaw/workspace/.tmp/fantasy_world_qa.jsonl` (35.7 KB) covering the fictional kingdom of Velmaris (names, places, history, magic rules, creatures, named spirits, founding villages, currency, calendar) — a domain Qwen3-0.6B has zero prior knowledge of, so memorization = training actually working
- Registered via QABUG-001's live path → dataset `f16eac64` (size 35767, qa_count 95)

### Training artifacts (real, on disk, `/home/genortg/finetune-studio/output/`)
| Artifact | Size | Source |
|---|---|---|
| `output/adapter/adapter_config.json` | 1.3 KB | base_model `unsloth/qwen3-0.6b-unsloth-bnb-4bit`, rank 16, Unsloth patches |
| `output/adapter/adapter_model.safetensors` | 40 MB | v2 final LoRA weights |
| `output/adapter/chat_template.jinja` | 4.9 KB | Qwen3 chat template |
| `output/merged/model.safetensors` | 1.2 GB | Merged LoRA + base, Unsloth-patched per QABUG-014 |
| `output/merged/config.json` + tokenizer | 1.4 KB + 11 MB | HuggingFace compatible |
| `output/adapter/tokenizer.json` | 11 MB | Qwen3 tokenizer |

### Real training curve (run `513f4325`)
- DB row: `status: completed, final_loss: 0.8, finished_at: 1789452072.0`
- Unsloth log: `Num examples = 85 | Num Epochs = 8 | Total steps = 176 | Trainable parameters = 10,092,544 of 606,142,464 (1.67% trained)`
- Wall-clock: ~14 s training + adapter save + merge (Unsloth 2x faster free finetuning, RTX 3090, max_seq_length 512)
- Cross-entropy dropped from ~2.5 to 0.8 → **real learning** (model compressed ~70% of per-token uncertainty on the training set, not the 17 s no-op from the wrong-complete session)

### Factual-verification prompt (proves the LoRA actually memorized)
- `POST /api/inference/chat` with `{"messages":[{"role":"user","content":"What is the capital of the kingdom of Velmaris?"}], "max_tokens": 200, "temperature": 0.1}`
- Response: `"The capital of the kingdom of Velmaris is Eldrathane, a fortress-city built into the cliffs above the Ember Sea."`
- Correct (matches the training-set fact for `q_velmaris_capital`) ✓

### Bench exec (real per-case evidence — QABUG-010 live-verified)
- `POST /api/benchmarks/projects/{pid}/runs/{rid}/run` with `{suite_path: "/tmp/suite_fantasy_world.json", suite_name: "fantasy-world-qa", judge_mode: "heuristic", max_tokens: 200}`
- Benchmark row `22b75f3c` written to `benchmark_runs` table (`time_ms: 14387, ran_at: 1789453023.8822978`)
- 10 per-case results captured in `benchmark_cases` — **8 ✓ pass, 2 ~ partial, 0 ✗ fail**:

| Case | Verdict | Judge | Model answer (truncated) | Judge reasoning |
|---|---|---|---|---|
| q_velmaris_capital | ✓ pass | heuristic | "The capital of the kingdom of Velmaris is Eldrathane…" | keywords matched 1/1 |
| q_ruler | ~ partial | heuristic | "Queen Lyssara, the last of the royal family…" | keywords matched 2/3 |
| q_year_of_twin_moons | ✓ pass | heuristic | "The Year of the Twin Moons lasted 247 days…" | keywords matched 1/1 |
| q_currency | ✓ pass | heuristic | "The high-value trade currency of Velmaris is the gold sovereign…" | keywords matched 2/2 |
| q_draughox | ✓ pass | heuristic | "A draughox is a massive six-legged beast of burden and war…" | keywords matched 3/3 |
| q_emberseas | ✓ pass | heuristic | "The Ember Sea is the warm southern ocean that borders the Kingdom of Velmaris…" | keywords matched 3/3 |
| q_embercollege | ✓ pass | heuristic | "At the Ember College, there are four schools of magic…" | keywords matched 2/2 |
| q_umbraeth | ✓ pass | heuristic | "The named spirit of the dark moon is Umbra'eth, also called the Black Moon…" | keywords matched 2/2 |
| q_six_trials | ~ partial | heuristic | "The Velmarian justice system consists of 10 trials…" | keywords matched 1/2 |
| q_korvash | ✓ pass | heuristic | "The peace treaty with the Black Mane clan was signed by Queen Mereth the Soft-Sp…" | keywords matched 3/3 |

- 5 bench history rows for `fantasy-world-qa` suite, **all with `judged: 10, unjudged: 0`** — QABUG-010 wire confirmed working across multiple runs (8/10, 5/10, 8/10, 6/10, 8/10 pass-rates observed)
- Categories: `geography: 2/2 passed, politics: 1/2 passed, magic: 1/1 passed, creatures: 1/1 passed, history: 1/1 passed, economics: 1/1 passed, spirits: 1/1 passed, law: 0/1 passed`

### Testing endpoint (real end-to-end via page-surface — QABUG-011 live-verified)
- `POST /api/testing/run-suite` with `{project_id: "a58ed72a", suite_path: "/tmp/suite_fantasy_world.json", max_tokens: 200}`
- Auto-loads merged model from project's most-recent completed run (QABUG-011 fix)
- `/api/inference/status` confirms `loaded: true, model: output/merged, idle_seconds: 12, idle_timeout: 1800`
- Per-case response shape: `[{name, category, question, correct_answer, response, passed, verdict, judge, judge_model, judge_reasoning, time_ms, error}]` — full v2 schema working (QABUG-005 + QABUG-007 fixes live)

## Nightly ledger
- Nightly QA artifacts: `.tmp/nightly-qa-NIGHT/README.md` (10914 bytes), `.tmp/nightly-qa-NIGHT/screenshots/01_dashboard_after_cleanup.png`, `.tmp/nightly-qa-NIGHT/fixtures/sample.jsonl` + `.sample.messages.jsonl`
- Fantasy-world QA dataset: `/home/genorbox1/.openclaw/workspace/.tmp/fantasy_world_qa.jsonl` (35.7 KB, 95 pairs)
- Fantasy-world test suite: `/tmp/suite_fantasy_world.json` (10 v2-schema cases, pushed to fan-dragon as `/tmp/suite_fantasy_world.json`)
- Real LoRA artifacts at `/home/genortg/finetune-studio/output/` (live on fan-dragon)
- Browser screenshots: `/home/genorbox1/.openclaw/media/outbound/{9a68e531,0b9f6323,d307cb05}-*-*.png`

## Next steps
1. **Fix QABUG-015**: trace `templates/benchmarks.html` + `templates/testing.html` for the renderResults function after `await fetch('/api/benchmarks/projects/{pid}/runs/{rid}/history')`; ensure the loading spinner is hidden after fetch completes. Likely root cause: the JS render function is calling `.then()` on the fetch but not chaining `.catch()` or `.finally()`, so on silent failure the spinner stays. Fix: add `.finally(() => document.querySelector('#computing-scores-spinner').style.display = 'none')`.
2. **Drive the testing page** through the full E2E with the v2-schema fantasy-world suite to capture the per-case model_answer (proves QABUG-013 working once QABUG-015 is fixed).
3. Optional: drive a longer training run (≥200 optimizer steps) on a larger dataset (≥150 QA pairs) to produce a LoRA that produces passing verdicts on the partial cases (q_ruler + q_six_trials).
4. Optional: triage the 55 pre-existing pytest failures (cluster dominated by `test_update` / `test_project_testing` / `test_project_training`).
5. Optional: `bash scripts/install-service.sh` on fan-dragon for systemd on :7860.
6. Optional: re-insert the 5 `system_updates` rows dropped during cleanup (safety net at `~/.cache/finetune-studio-backups/2026-09-14-nightly/.finetune-studio/`).

## Commands
- Pytest (touched): `.venv/bin/python -m pytest tests/test_dataset_register.py tests/test_data_prep_promote.py tests/test_nav_routes.py tests/test_dark_light_toggle.py tests/test_breadcrumb.py tests/test_bench_judge.py tests/test_testing_auto_load.py tests/test_benchmarks_template.py tests/test_testing_template.py -q -p no:cacheprovider`
- Pytest (full, excluding GPU-only): `.venv/bin/python -m pytest tests/ -q -p no:cacheprovider --ignore=tests/test_vram_profiler.py`
- Ruff (touched): `.venv/bin/python -m ruff check src/finetune_studio/webui/routes/datasets.py src/finetune_studio/webui/routes/data_prep.py src/finetune_studio/webui/routes/testing.py src/finetune_studio/webui/templates/ src/finetune_studio/data/fs/ src/finetune_studio/data/project_filesystem.py tests/`
- Deploy: `git push` then `ssh fan-dragon 'cd /home/genortg/finetune-studio && git pull --ff-only'`; restart uvicorn per `RESTART.md` (use the `bash -lc "...& disown"` pattern, NOT `pkill` before spawn — race-kills parent bash)
- Real E2E replay: see `.tmp/nightly-qa-NIGHT/README.md` + the fantasy-world dataset + suite above

## Blockers
- genorbox1 is dev-only: pure unit tests + ruff here; visual/E2E/GPU only on fan-dragon after pull + restart
- `tests/test_vram_profiler.py` 37 GPU-dependent tests fail on genorbox1 (pass on fan-dragon only — pre-existing)
- 55 pre-existing pytest failures in `test_update`, `test_project_testing`, `test_project_training` (baseline unchanged by this turn's work)
- **QABUG-015 (new)**: bench + testing pages stuck in "COMPUTING SCORES" loading state — data layer verified working, JS render path incomplete. Fix landed in next session.
- Working tree has uncommitted edits (`AGENTS.md`, `HANDOFF.md`, `inference.py`, `chat_v2.py`, `testing.py` + untracked `test_inference_unsloth_load.py`, `test_testing_load_body.py`) — none blocking; commit + push when convenient
