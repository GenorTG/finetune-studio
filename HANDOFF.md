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
| DPO E2E | Browser upload/health: 16/16 usable, 2 held out. One-epoch real DPO run completed 14/14 steps (35 s; final loss 0.6757, eval loss 0.6417), no merge. Pipeline smoke only, not a quality claim. |
| Agent E2E | UI loaded helper at 32768 context, full 48/48 layers; prep returned 2 accepted/1 rejected. Agent listed/read a source, created 2 grounded Q&A pairs, and the Pairs page showed both pending. Final readiness reply matched tool data: 2 sources/2 parsed; 5 approved, 11 pending, 0 rejected; 1 dataset, 0 RAG; next: review pending pairs. |
| Preference authoring | Branch `feat/pref-data` (not merged/deployed): `data/prep/preference.py`, `POST /api/projects/{pid}/data-prep/preference` (+ `/progress`), `fts dataset build-preference`, Pairs-page card, Training deep link. CPU tests + sandbox UI (fake helper) verified; **never run against the real helper/GPU yet**. |
| Cleanup | Disposable project `e2e-dpo-route-1006` (`9c1759cd`) deleted via UI; API returns 404; project/output/data dirs and all three temporary upload fixtures are absent. Helper unloaded. |

## Completed foundations

- Q8_0 llama.cpp abort fixed with `n_ubatch=256`; context is not reduced. GGUF loader autofits GPU layers/CPU fallback, verified on the 3090.
- Compute-device selection, wizard/loader/error fixes, dataset-build CLI, CI rework, and earlier full user-flow walkthrough are merged.
- DPO upload preserves preference rows; dataset health follows the selected training route. Helper autoload now retries after inference idle-unload. Readiness counts parsed sources by nonzero `chunk_count`, and final readiness replies use the authoritative tool summary.
- `9021cfb` added DPO, tool-call SFT, raw-text continued pretraining, reviewed reasoning distillation, agent guide/readiness tools, and route standards in `tests/E2E_MANUAL_GUIDE.md`.
- Preference authoring (hallucination + abstain kinds): rejected answers come only from the helper; gates in `preference.py`; output passes `format_for_preference` unchanged and registers like an export (`source=data-prep-preference`).
- ORPO/KTO are unsupported by installed TRL 1.14.1; do not relabel or substitute DPO.

## Next steps

1. Merge `feat/pref-data`, restart the user unit, then on a throwaway `e2e-*` project: load the helper, build preference pairs, spot-check 10, run a 1-epoch DPO from the card's link, and compare hallucination/abstention on a held-out suite before/after (the real quality proof is still missing).
2. Diagnose the official GSM8K UI run before attempting its full 1,319 cases; a prior browser wait exceeded 300 s. Use bounded samples and visible progress first.
3. Decide whether to require `ci-ok` in `main` branch protection.
4. Deploy to fan-dragon only when Genor asks; deployment remains deferred.

## Commands

- Manual user walkthrough and route standards: `tests/E2E_MANUAL_GUIDE.md`.
- Browser phases: `FTS_ALLOW_LIVE_E2E=1 .venv/bin/python tests/e2e_user_walkthrough.py --list`.
- Focused tests: `.venv/bin/python -m pytest tests/<file>.py -q -p no:cacheprovider`; lint: `.venv/bin/ruff check src/ scripts/`.
- Install: `bash install.sh --verify`; `.venv/bin/python scripts/install_diagnose.py --check --no-service-check`.

## Known issues

- DPO smoke logged TRL Qwen tokenizer prefix-mismatch warnings; training completed, but the 16-row run is too small to establish behavior quality.
- Preference pairs are authored in bulk with gates, not hand-reviewed per pair (no review UI); `style` pairs, grounded-prompt (RAG-layout) preference rows, and an on-policy rejected model (the model being tuned) are not implemented. ORPO/KTO, automatic teacher-trace generation, and eval decontamination are not implemented.
- Official GSM8K full run remains unverified; synthetic smoke is not an official score.
