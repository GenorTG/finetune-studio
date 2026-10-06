# Handoff — finetune-studio

## Mission

Local fine-tune, data-prep and RAG WebUI. GPU-first on every vendor, CPU only on GPU-less hosts, never a silent fallback (Genor 2026-10-05). Genor allows commits/pushes to origin, service restarts and real-hardware tests.

## State (verified 2026-10-06 ~14:40)

| Area | State |
|---|---|
| Git / CI | `main` = `origin/main`, tree clean. CI = SHA-pinned actions, lint+codemap, 5 CPU shards, `install.sh --plan` smoke (ubuntu+macOS), `ci-ok` aggregate; Dependabot weekly. Last pushed run green before this session's final commits — check `gh run list`. |
| Service | genorbox1 :7860, user unit, RTX 3090 only. Restarted on the merged build (loader autofit, GPU-select API, all fixes below). |
| Local tests | Full suite run in chunks this session on intermediate trees: every chunk green except `test_repo_hygiene` while new files were untracked (fixed by committing). Re-run all 5 chunks before the next release. |
| Manual E2E | `tests/E2E_MANUAL_GUIDE.md` (step-by-step human guide) + `tests/e2e_user_walkthrough.py` (same path automated, real browser, 13 phases, background GPU/helper/journal monitor). Evidence/screens `.tmp/qa-shots/manual-e2e/`, logs `.tmp/manual-e2e/full*.log`. |

## Done this session

- **llama.cpp abort fixed at the root** (not OOM): CUDA kernel bug on a full 512-token micro-batch for Q8_0 → loader caps `n_ubatch` at 256. Loader now **keeps n_ctx** and autofits GPU layers (`models/gguf_fit.py`, native-log capture `models/llama_native_log.py`, CPU-only last attempt); verified under real VRAM pressure (12B @32k: 14/48 layers at 6 GiB free, CPU-only at 0.3 GiB). Inference page slider is live; load/status endpoints report placement.
- **In-app GPU selection** (lane, merged): persisted Auto/one GPU/CPU choice applied at next start; `GET/PUT /api/system/compute-device`; Settings card.
- **Found by the user-style walkthrough and fixed**: helper load froze the whole app 20–56 s (async handler + manager lock); chunk overlap cut words; HTML title duplicated; install diagnostics ignored the `[parsers]` extra (host lacked xlrd/pptx/bs4/striprtf — repaired); file-library type labels; abbreviation-aware sentence split (`Dr.`); standard training path never generated the quiz and the child recorded it without a run id; Testing page now splits from-memory vs with-context; contrast/size findings.
- `fts dataset build`, CI rework, wizard export-gate buttons, quiz asks grounded rows with their context (earlier in the session).

## Real numbers (0.6B, 14 mixed-format files, Precision preset 420 steps ≈ 6.5 min on the 3090)

Parse 14/14 formats in 3 s · helper mining 14 files ≈ 100 s, 12/12 ground-truth facts surfaced · 61 approved pairs (24 with retrieved context) · loss 0.035 · quiz 59/61 = 96.7 % (memory 35/37, with context 24/24) · **held-out slice 3/7 then 5/7 (42.9 % / 71.4 % on two identical runs — n=7 is noise)** · pure recall without retrieval 7/12, with RAG 11/12 · offline MMLU-shaped smoke 58–75 % (48 cases, noisy, not industry).

## Open — needs Genor (the native question tool was unavailable this session)

1. **New data/training routes** (app is SFT-on-QA + RAG only): DPO/ORPO preference tuning, tool-calling/agentic SFT, continued pre-training, reasoning distillation — which first? Recommendation: preference (abstention) + tool-calling.
2. **Agent mode** has 4 data-prep tools only (`list_sources`, `read_source`, `list_qa_pairs`, `create_qa_pairs`); it cannot navigate, approve, build RAG, export, train or test. Build whole-app control with live UI feedback?
3. Guide location chosen: tracked `tests/E2E_MANUAL_GUIDE.md` (say if it should be local-only).

## Next steps

1. Answer the questions above; then build the chosen routes (design in the guide's routes table).
2. Pure recall is weak (7/12): add paraphrase augmentation (≥3 phrasings per fact) to prep and measure; the held-out slice (7 rows) is the honest generalisation signal but far too small — grow it.
3. Official-suite run on the base row of Benchmarks hung >4 min in the UI (GSM8K sampled 50) — not diagnosed; reproduce with the API and show progress/errors.
4. Chat page silently attaches RAG in "test" mode, so "recall" tests are grounded; add an explicit RAG toggle.
5. Chat page: after the Testing step leaves the merged model resident, the inline model dropdown is empty/hidden (switch only via header chip or /inference) — reproduce and fix. Benchmarks case-name column breaks words mid-word.
6. `llama-cpp-python` 0.3.36 is source-built; try the abetlen cu13x wheel and remove the ubatch cap if fixed upstream (`FTS_LLAMA_UBATCH`).
7. Branch protection: require `ci-ok`. AMD/Intel/Apple stay plan-tested only; fan-dragon deploy deferred.

## Commands

- Walkthrough: `FTS_ALLOW_LIVE_E2E=1 .venv/bin/python tests/e2e_user_walkthrough.py --list` then `--phase create --phase upload …` (cleanup phase deletes the throwaway project + base model).
- Tests: chunked, ≤10 min per call — `ls tests/test_*.py | grep -v test_vram.py | awk -v k=0 'NR%5==k'` (k=0..4). Lint `.venv/bin/ruff check src/ scripts/`; `make codemap-check`.
- GPU/plan: `bash install.sh --check`, `.venv/bin/fts accel`; dataset: `fts dataset build --project <name|id> [--force] [--json]`.

## Blockers

- None.
