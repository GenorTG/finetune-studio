# Handoff — finetune-studio

## Mission

Local fine-tune, data-prep and RAG WebUI. GPU-first; never silently fall back (Genor 2026-10-05).
Commits, pushes, local service restarts and real-hardware tests are authorized.

## State (verified 2026-10-06)

| Area | State |
|---|---|
| Git | Base `main`/ `origin/main` = `275ff3f`; data-route continuation is staged for commit after focused verification. |
| CI | Current `main` is green. SHA-pinned Actions, 5 test shards, `ci-ok`, Ubuntu/macOS install-plan smoke, weekly grouped Dependabot. |
| Service | genorbox1 :7860 active on previous commit. App targets RTX 3090; GTX 1070 is not to be disturbed. Restart after pushing the tested SHA. |
| Verification | 43 focused tests pass; `ruff check src/ scripts/`, `uv pip check`, and `git diff --check` pass. Installer reports no issues; llama.cpp CLI now built with CUDA. |
| E2E | The prior whole-app walkthrough completed except Chat options were read before async model discovery. Fresh browser validation on this continuation is pending. |

## Completed foundations

- Q8_0 llama.cpp abort fixed at the native cause: `n_ubatch=256`; context is never reduced. GGUF loader autofits GPU layers and falls back to CPU-only, verified on the 3090.
- Compute-device selection, wizard/loader/error fixes, dataset-build CLI, CI rework, grounded-row quiz, coverage-gate recovery and parser diagnostics are merged.
- Existing walkthrough: 14 mixed formats parsed; helper surfaced 12/12 facts; 61 approved pairs; SFT trained 420 steps; quiz 59/61 (96.7%). Held-out 3/7 then 5/7 is too small to call a benchmark.
- Current changes add DPO via TRL 1.14.1; tool-call SFT with schemas; raw-text continued pretraining; reviewed reasoning distillation; agent guide/readiness tools; industry-route and browser walkthrough guidance.
- ORPO/KTO are explicitly unsupported by the installed TRL runtime; do not substitute DPO under those labels.

## Next steps

1. Commit and push after the focused checks; verify GitHub Actions.
2. Restart local :7860 on that SHA. Use the browser for a real supported-route run, Chat model discovery/load, and agent guide/readiness feedback; watch GPU0/journal.
3. Diagnose the official GSM8K UI timeout from its API/page path; do not rerun all 1,319 cases without visible progress/cancellation. Synthetic MMLU is not an official score.
4. Record live evidence here, clean up only the throwaway project/data, and verify the repo is clean.

## Commands

- Human E2E guide: `tests/E2E_MANUAL_GUIDE.md`.
- Walkthrough phases: `FTS_ALLOW_LIVE_E2E=1 .venv/bin/python tests/e2e_user_walkthrough.py --list`.
- Focused checks: `.venv/bin/python -m pytest tests/<file>.py -q -p no:cacheprovider`; lint `.venv/bin/ruff check src/ scripts/`.
- Install: `bash install.sh --verify`; `.venv/bin/python scripts/install_diagnose.py --check --no-service-check`.
- Full CPU suite: five chunks excluding `tests/test_vram.py` (see repo `AGENTS.md`).

## Known issues

- ORPO/KTO, in-app preference comparison authoring and automatic teacher-trace generation are not implemented.
- Official GSM8K run remains unverified; its previous UI wait expired after 300 s.
- fan-dragon deployment remains deferred.
