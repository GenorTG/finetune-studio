# Handoff — finetune-studio

## Mission

Local fine-tune, data-prep and RAG WebUI. GPU-first; never silently fall back.
Commits, pushes, local service restarts and real-hardware tests are authorized.

## State (verified 2026-10-07)

| Area | State |
|---|---|
| Git | `main` has the three lanes merged (`feat/pref-train`, `feat/pref-data`, `feat/app-guide`; merge commits `2f2a1f5`, `a1984fe`, `7bfc678`) plus `054c3e9` (4-bit-up-front fix). Branch tips still on origin; side worktrees removed. |
| Tests | Merged tree before the fix: 5 chunks, 2362 passed / 1 skipped; ruff + codemap clean. After the fix: `test_training_4bit_fit.py` + accel wiring files pass (80). |
| Service | genorbox1 :7860 runs the merged build with the fix (`/api/system/version` 0.1.0.209); 3090 only, idle ~1.2 GiB. |
| Walkthrough (live, real browser) | `e2e_user_walkthrough.py` all phases green: create/upload/prep/review 22/22, rag/export/model 10/10, train/gguf 10/10 (SFT Qwen3-0.6B 420 steps, loss 0.038), chat/test/bench 10/10 (12/12 recall, quiz 93.4 %, held-out 28.6 % = 2/7 — memorisation, not generalisation), cleanup 6/6. |
| Pair authoring (browser tool, real Gemma-12B) | Pairs page card built 30 pairs (15 hallucination + 15 abstain), dataset registered, "Train with this dataset" link works. Hallucination pairs are sound (chosen = source fact, rejected = confident invention); abstain pairs plausible. Chosen/rejected length ratio 0.13 (chosen ~67 chars vs ~524) — the report only warns at ≥1.3, so a length bias toward short answers is not flagged. |
| DPO (browser tool, Qwen3.5-9B) | First try OOM'd in the first forward pass (bf16 load = 18 GiB succeeded, so the load-time ladder never fired). After `054c3e9` the run loaded 4-bit, 42/42 steps in ~90 s, peak 15.7 GiB, reward accuracy 1.0, margin ~13, **no TRL prefix-mismatch warnings** (pref-train fix confirmed on the real tokenizer). Pipeline proof only: 30 near-separable rows, no before/after quality eval on 9B. |
| Guide (browser tool, real helper) | Panel opens on every page; one question produced `project_overview` → `navigate` → `suggest_settings` tool cards, the page changed to Training and pre-filled route/epochs/rank, final answer told the user to press Start. Helper cold-load inside the stream took ~80 s. |
| Cleanup | Throwaway project `ux-walk-1` (`cbd39e4f`) deleted via UI, base model download removed, `projects`/`output` empty; helper unloaded. DB backup in `.tmp/db-backup/`. |

## Done this session

- Merged the three lanes (conflicts were only `HANDOFF.md`, `AGENTS.md` gotchas, `project_training.html` DPO help text, `app.py` router imports — kept both sides).
- `models/hf_loader.training_needs_4bit` + `engine._load_model_with_fallback`: when bf16 weights + 6 GiB headroom exceed free VRAM and bitsandbytes works, load 4-bit before training and say so in the live panel/log.

## Next steps

1. Guide answer quality: it picked DPO for a project with 61 approved pairs and no SFT run — DPO from a raw base is the wrong first step. Tune `guide/prompt.py` / `guide/kb/training*.md` (recommend SFT first, DPO on the SFT-merged run) and re-ask via the browser; questions in `tests/E2E_MANUAL_GUIDE.md` §12.
2. Preference quality: run `build-preference` + DPO on the SFT-merged run of a real project and measure in-corpus accuracy / refusal rate (`scripts/pref_quality_eval.py`) at 9B; add a length-ratio warning for chosen ≪ rejected.
3. Responsiveness: the walkthrough monitor logged one `api? TimeoutError` during GGUF export (walk3 log) — check whether the event loop blocks during conversion.
4. Guide helper cold-load (~80 s) shows no progress in the panel until the stream starts; confirm the "loading helper" status renders immediately.
5. Diagnose the official GSM8K UI run with bounded samples before the full 1,319 cases.
6. Decide whether `main` protection should require `ci-ok`. fan-dragon deploy stays deferred.

## Commands

- Manual walkthrough and route standards: `tests/E2E_MANUAL_GUIDE.md`; phases: `FTS_ALLOW_LIVE_E2E=1 .venv/bin/python tests/e2e_user_walkthrough.py --list` (order: create upload prep review rag export model train gguf chat test bench cleanup).
- Focused tests: `.venv/bin/python -m pytest tests/<file>.py -q -p no:cacheprovider`; full suite = 5 chunks (see AGENTS gotcha), ~18 min each when run in parallel; lint `.venv/bin/ruff check src/ scripts/`.
- Guide sandbox proof: `PYTHONPATH=src .venv/bin/python tests/e2e_guide_sandbox.py --port 7896`.
- Install: `bash install.sh --verify`; `.venv/bin/python scripts/install_diagnose.py --check --no-service-check`.

## Known issues

- DPO/KTO quality evidence is from Qwen3-0.6B at tiny n: DPO + keep-chosen moves behaviour (abstention 0 → up to 23/24) but costs 5–16/60 facts and over-refuses; KTO did not help. Treat preference tuning as opt-in.
- ORPO is not wired (`trl.experimental.orpo` only). Automatic teacher traces and eval decontamination are not implemented.
- Stale, not from this work: `projects/6a7e1b92/curated.db` (Sep 7, not in the API) and `.tmp/rag-repro` (4.4 GB) — left for Genor to decide.
- spa.js re-entry of full-loaded pages falls back to a full reload (AGENTS gotcha) and would cut an in-flight Guide stream.
