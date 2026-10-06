# Handoff — finetune-studio

## Mission

Local fine-tune, data-prep and RAG WebUI. GPU-first; never silently fall back.
Commits, pushes, local service restarts and real-hardware tests are authorized.

## State (verified 2026-10-06)

| Area | State |
|---|---|
| Git | `ab5d846` is repo HEAD; the active app code was deployed from `87497e6` (later commit only updated HANDOFF and hook-generated `VERSION`). Working tree clean. |
| CI | `37500532291`: 5/5 test shards, ruff/codemap, Ubuntu/macOS plan smoke, and `ci-ok` green. Pages run `37500530615` green. `main` branch protection still does not require `ci-ok`. |
| Service | genorbox1 :7860 active on `87497e6`; `/api/projects` HTTP 200; inference unloaded. RTX 3090 idle at ~1.2 GiB; leave the GTX 1070 foreign workload alone. |
| DPO E2E (service, `87497e6`) | Browser upload/health: 16/16 usable, 2 held out. One-epoch DPO run 14/14 steps (loss 0.6757 ≈ ln 2: the 1e-6 LR did not move the adapter). Pipeline smoke only. |
| Preference branch `feat/pref-train` | DPO hardened + KTO route + start-from-merged-run + live reward metrics; **not on main / not deployed**. Evidence below. |
| Agent E2E | UI loaded helper at 32768 context, full 48/48 layers; prep returned 2 accepted/1 rejected. Agent listed/read a source, created 2 grounded Q&A pairs, and the Pairs page showed both pending. Final readiness reply matched tool data: 2 sources/2 parsed; 5 approved, 11 pending, 0 rejected; 1 dataset, 0 RAG; next: review pending pairs. |
| Cleanup | Disposable project `e2e-dpo-route-1006` (`9c1759cd`) deleted via UI; API returns 404; project/output/data dirs and all three temporary upload fixtures are absent. Helper unloaded. |

## In flight

Nothing running. `feat/pref-train` is pushed, unmerged.

## Completed foundations

- Q8_0 llama.cpp abort fixed with `n_ubatch=256`; context is not reduced. GGUF loader autofits GPU layers/CPU fallback, verified on the 3090.
- Compute-device selection, wizard/loader/error fixes, dataset-build CLI, CI rework, and earlier full user-flow walkthrough are merged.
- DPO upload preserves preference rows; dataset health follows the selected training route. Helper autoload now retries after inference idle-unload. Readiness counts parsed sources by nonzero `chunk_count`, and final readiness replies use the authoritative tool summary.
- `9021cfb` added DPO, tool-call SFT, raw-text continued pretraining, reviewed reasoning distillation, agent guide/readiness tools, and route standards in `tests/E2E_MANUAL_GUIDE.md`.
- TRL 1.14.1: `KTOTrainer` is stable API (the old "ORPO/KTO unsupported" line was wrong for KTO); ORPO exists only as `trl.experimental.orpo` and is not wired. Never relabel DPO as either.

## Preference quality (feat/pref-train, RTX 3090, Qwen3-0.6B, `scripts/pref_quality_eval.py`)

Setup: fictional corpus (6 orders × 5 facts); SFT (LoRA r32, 6 ep, 50 s) on 3 phrasings/fact, merged; then DPO/KTO from that
merged model on 60 fact pairs (chosen = source answer, rejected = confident wrong value) + 20 abstain pairs (chosen =
"That isn't covered in the provided documents."). Greedy, thinking off, same system prompt everywhere. Held-out: 60 new
phrasings of documented facts; 24 new phrasings of the *trained* undocumented attributes; 30 novel undocumented attributes.

| Model | in-corpus correct | false refusals | refuses seen-attr (n=24) | refuses novel-attr (n=30) |
|---|---|---|---|---|
| base Qwen3-0.6B | 0/60 | 56 | 24 | 22 |
| SFT (start point) | 59/60 | 0 | 0 | 0 |
| DPO plain, lr 5e-6 (old default ≈ TRL) | 59/60 | 0 | 0 | 0 — adapter did not move |
| DPO plain, lr 3e-5 | 31–48/60 (2 runs) | 0 | 0 | 0 — forgot facts |
| **DPO + keep-chosen 1.0, lr 1e-4, 3 ep** seed 11 / seed 12 | 54/60 / 50/60 | 2 / 2 | 6 / 16 | 3 / 6 |
| DPO + keep-chosen, 36 pairs + 24 abstain, 5 ep | 43/60 | 10 | 23 | 10 |
| KTO lr 5e-5 / lr 1e-5 | 51/60 / 54/60 | 0 / 0 | 0 / 0 | 0 / 0 |

Verdict: the machinery works and behaviour changes in the intended direction (abstention 0 → up to 23/24 on trained
attributes), but it trades facts for it (59 → 43–54/60) and generalises weakly to unseen attributes (≤ 10/30). n is tiny, seeds
differ a lot (6 vs 16 refusals), nothing here is a significance claim. Plain DPO on LoRA either does nothing (≤ 5e-6) or
forgets (≥ 3e-5); the keep-chosen NLL term (`preference_sft_weight`, TRL `loss_type=["sigmoid","sft"]`) is what makes it usable.
KTO ran cleanly (108 steps, margin ≈ 2.7, KL ≈ 0.5) but did not help here. Cost: DPO train ≈ 50 s, KTO ≈ 100 s, whole
SFT→DPO→3 evals 116 s; GPU0 peak 4.9 GiB gross (≈ 3.7 GiB over the 1.2 GiB foreign baseline). GPU0 back at 1207 MiB.

- Prefix-mismatch cause: Qwen3.5 template (open `<think>\n` generation prompt vs `<think>\n\n</think>` rendered turn; BPE merges
  `\n`+`\n`). Qwen3-0.6B never mismatched. `training/preference_tokens.py` renders prompts with `enable_thinking=False`.
- Defaults (`PREFERENCE_DEFAULTS`, routes/training.py): DPO lr 1e-4 / 3 ep / keep-chosen 1.0; KTO lr 1e-5 / 3 ep. Measured on ~50-step
  runs only: the route warns above 300 steps at lr ≥ 5e-5; Zephyr DPO-QLoRA uses 5e-6 over ~15k steps.

## Next steps

1. Merge `feat/pref-train` after review, then `systemctl --user restart finetune-studio` and run the DPO/KTO pages once in the browser (not done: no live service touched here).
2. Try an abstain-aware recipe that does not over-refuse (more abstain pairs than facts hurt recall: 43/60); needs a larger corpus than 30 facts to mean anything.
3. Diagnose the official GSM8K UI run before attempting its full 1,319 cases; use bounded samples first.
4. Decide whether to require `ci-ok` in `main` branch protection.
5. Deploy to fan-dragon only when Genor asks; deployment remains deferred.

## Commands

- Manual user walkthrough and route standards: `tests/E2E_MANUAL_GUIDE.md`.
- Browser phases: `FTS_ALLOW_LIVE_E2E=1 .venv/bin/python tests/e2e_user_walkthrough.py --list`.
- Focused tests: `.venv/bin/python -m pytest tests/<file>.py -q -p no:cacheprovider`; lint: `.venv/bin/ruff check src/ scripts/`.
- Install: `bash install.sh --verify`; `.venv/bin/python scripts/install_diagnose.py --check --no-service-check`.

## Known issues

- Preset-based starts (`preset_id`) never set `config.data_path`/`project_id`, so the post-train quiz is skipped for them (SFT too); the form path sets both.
- ORPO (experimental TRL), preference-comparison authoring, teacher-trace generation and eval decontamination are not implemented.
- Official GSM8K full run remains unverified; synthetic smoke is not an official score.

## Blockers

None.
