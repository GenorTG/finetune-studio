# Handoff — finetune-studio

## Mission

Local fine-tune, data-prep and RAG WebUI. GPU-first; never silently fall back.
Commits, pushes, local service restarts and real-hardware tests are authorized.

## State (verified 2026-10-06)

| Area | State |
|---|---|
| Git | `13156a2` is pushed; current DPO dataset-health fix is tested locally, pending commit/push. |
| CI | `9021cfb`: all checks green (5 test shards, ruff/codemap, Ubuntu/macOS plan smoke, Pages). Main remains unprotected; `ci-ok` is not required by a branch rule. |
| Service | genorbox1 :7860 active on `13156a2`; HTTP 200; selected compute is RTX 3090. GTX 1070 remains a foreign workload; browser rendering may use it, do not load models there. |
| Verification | 44 focused dataset/preference tests passed; repo Ruff and `git diff --check` pass. Earlier install verification/CUDA CLI build remain clean. |
| E2E | Disposable project `e2e-dpo-route-1006` (`9c1759cd`). Browser upload of valid DPO JSONL now returns HTTP 200 and registers 16 rows. Found and fixed the SFT-only health check falsely reporting DPO rows as untrainable. Needs live UI retest after next push/restart, then clean up. |

## Completed foundations

- Q8_0 llama.cpp abort fixed with `n_ubatch=256`; context is not reduced. GGUF loader autofits GPU layers/CPU fallback, verified on the 3090.
- Compute-device selection, wizard/loader/error fixes, dataset-build CLI, CI rework and previous full user-flow walkthrough are merged.
- `9021cfb` adds DPO, tool-call SFT, raw-text continued pretraining, reviewed reasoning distillation, agent guide/readiness tools, and industry-route documentation.
- ORPO/KTO are unsupported by installed TRL 1.14.1; never relabel or substitute DPO.

## Next steps

1. Commit/push the tested DPO health fix and verify all Actions.
2. Restart local :7860 on that SHA; check DPO health shows 16/16 trainable, then run a short one-epoch DPO smoke through the browser and confirm final status.
3. Verify Chat model discovery and agent guide/readiness with browser-visible feedback; inspect console/network and GPU/journal.
4. Delete the throwaway project and temporary upload fixture; update this handoff with live results.
5. Diagnose why the official GSM8K UI run appeared stuck; bound sampling/progress before running all 1,319 cases. Synthetic smoke is not an official score.

## Commands

- Manual user walkthrough: `tests/E2E_MANUAL_GUIDE.md`.
- Browser phases: `FTS_ALLOW_LIVE_E2E=1 .venv/bin/python tests/e2e_user_walkthrough.py --list`.
- Focused checks: `.venv/bin/python -m pytest tests/<file>.py -q -p no:cacheprovider`; lint `.venv/bin/ruff check src/ scripts/`.
- Install: `bash install.sh --verify`; `.venv/bin/python scripts/install_diagnose.py --check --no-service-check`.

## Known issues

- DPO comparison authoring, ORPO/KTO, and automatic teacher-trace generation are not implemented.
- Official GSM8K full run is unverified; previous browser wait expired after 300 s.
- fan-dragon deployment remains deferred.
