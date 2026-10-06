# Handoff — finetune-studio

## Mission

Local fine-tune, data-prep and RAG WebUI. GPU-first; never silently fall back.
Commits, pushes, local service restarts and real-hardware tests are authorized.

## State (verified 2026-10-06)

| Area | State |
|---|---|
| Git | Pushed base `9021cfb`; a follow-up fix for DPO upload conversion is tested locally, pending commit/push. |
| CI | `9021cfb`: all checks green (5 test shards, ruff/codemap, Ubuntu/macOS plan smoke, Pages). Main remains unprotected; `ci-ok` is not required by a branch rule. |
| Service | genorbox1 :7860 active on `9021cfb`; selected compute is RTX 3090. GTX 1070 remains a foreign workload; browser rendering may use it, do not load models there. |
| Verification | Route/data focused 43 passed; follow-up converter + DPO tests 28 passed; repo Ruff, uv pip, install verify, CUDA CLI build, and diff checks passed. |
| E2E | Created throwaway project `e2e-dpo-route-1006` (id `9c1759cd`). Browser exposed upload conversion rejecting valid prompt/chosen/rejected JSONL (HTTP 400); converter fix plus tests now pass. Retest UI upload/run after new commit reaches the service, then delete project. |

## Completed foundations

- Q8_0 llama.cpp abort fixed with `n_ubatch=256`; context is not reduced. GGUF loader autofits GPU layers/CPU fallback, verified on the 3090.
- Compute-device selection, wizard/loader/error fixes, dataset-build CLI, CI rework and previous full user-flow walkthrough are merged.
- `9021cfb` adds DPO, tool-call SFT, raw-text continued pretraining, reviewed reasoning distillation, agent guide/readiness tools, and industry-route documentation.
- ORPO/KTO are unsupported by installed TRL 1.14.1; never relabel or substitute DPO.

## Next steps

1. Commit/push the tested DPO upload converter fix and verify all Actions.
2. Restart local :7860 on that SHA. In the existing throwaway project, upload `/home/genorbox1/.openclaw/media/inbound/fts-dpo-route-smoke.jsonl`, then start a short one-epoch DPO smoke with visible progress; confirm final run status and cleanup.
3. Verify Chat model discovery/load and agent guide/readiness via the real browser; check console/network and GPU/journal.
4. Update handoff with live results; delete only the throwaway project and temporary fixture.
5. Diagnose why the prior official GSM8K UI run appeared stuck. Its 1,319 cases need bounded sampling/progress; synthetic smoke is not an official score.

## Commands

- Manual user walkthrough: `tests/E2E_MANUAL_GUIDE.md`.
- Browser phases: `FTS_ALLOW_LIVE_E2E=1 .venv/bin/python tests/e2e_user_walkthrough.py --list`.
- Focused checks: `.venv/bin/python -m pytest tests/<file>.py -q -p no:cacheprovider`; lint `.venv/bin/ruff check src/ scripts/`.
- Install: `bash install.sh --verify`; `.venv/bin/python scripts/install_diagnose.py --check --no-service-check`.

## Known issues

- DPO comparison authoring, ORPO/KTO, and automatic teacher-trace generation are not implemented.
- Official GSM8K full run is unverified; previous browser wait expired after 300 s.
- fan-dragon deployment remains deferred.
