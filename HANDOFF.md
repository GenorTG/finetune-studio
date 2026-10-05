# Handoff — finetune-studio

## Mission

Local fine-tune + data-prep + RAG WebUI. Current thread: GPU-acceleration overhaul — GPU first on every
vendor (NVIDIA/AMD/Intel/Apple), CPU only on GPU-less hosts, never a silent CPU fallback (Genor 2026-10-05).

## State (verified 2026-10-05)

| Area | State |
|---|---|
| Git | `main` @ `4ea403a`, **ahead of origin by 5** (HANDOFF fix + three GPU-acceleration commits + project-doc refresh; unpushed). The final suite started on code commit `8c54955`, before the docs-only commit. Push only with Genor's OK. |
| genorbox1 | Service active on :7860 (user unit, `FTS_ROOT=~/.finetune-studio`), VERSION 0.1.0.148, torch 2.14.1+cu130, CUDA OK; GPUs RTX 3090 (0) + GTX 1070 (1, never use). |
| Tests | Full suite on `8c54955`: 1704 passed, 1 skipped, 2 failed (version values raced a concurrent docs commit bumping VERSION .151→.152); rerun `tests/test_version.py`: 4 passed. Earlier lane failures: hygiene failure disappeared after commits; version-hook failure was caused by `FTS_NO_BUMP=1`. |
| Ruff | `ruff check src/ scripts/` clean on the working tree; `tests/` ~108 legacy findings. |
| Working product paths | Project → files → parse → pair gen → LoRA train (checkpoints/eval/early stop, Unsloth toggle) → merge/GGUF export → testing/benchmark (review, verdict override) → Chat with tuned model + RAG; encrypted RAG export/`.ftsrag` import; onboarding tour; idle unload (`FTS_IDLE_TIMEOUT`, default 300 s). E2E-verified on RTX 3090 2026-10-02/04. |
| UI coverage ledger | Slices 1-5 of `docs/audit/UI-COVERAGE-2026-10-03.md` done (training fields, dataset health, benchmark review, nav/wording, dead code); css `?v=75` in working tree. |
| Grounded export rows | `GET data-prep/export` + subset build take `grounded_share` (0-1; omitted = 40% when a RAG corpus exists, else off) and `distractors` (0-2); data-prep UI has the checkbox + % box. Layout is `data/rag_portable/prompt.py` (shared with `rag_chat`). Live 2026-10-05 (Qwen3-0.6B, 150 steps): rag/chat grounded 2/5 → 4/5 (n=1 run; distractors unmeasured live). Tests: `tests/test_grounded_rows.py`. No `fts` command builds datasets, so no CLI flag. |
| Test-file review | Incomplete: 8 of ~198 `test_*.py` read (`docs/audit/TEST-AUDIT-2026-10-01.md`). Do not claim complete. |

## In flight (code committed; verification complete)

- **Lane B — installers.** `scripts/accel_plan.py` (new, stdlib-only detect → plan → install; sole source of GPU
  stack choice) called by `install.sh`, `update.sh`, `install.ps1`, `install.bat`, `scripts/install_diagnose.py
  --repair` (rewritten GPU checks). `bash install.sh --plan` previews; `--gpu <vendor>`, `FTS_FORCE_VENDOR`,
  `FTS_ACCEL_FIXTURE`, `FTS_UNSLOTH=auto|1|0`. Docs updated: `docs/DEPLOYMENT.md`, `docs/DEPENDENCIES.md`.
  `pyproject.toml`/`requirements.txt`: transformers `>=5.0,<6`. Tests: `test_accel_plan.py`,
  `test_install_diagnose*.py`, `test_installer_contract.py`.
- **Lane C1 — training/loading/export.** New `src/finetune_studio/accel/` (`device`, `env` = `FTS_GPU_DEVICES`/
  `FTS_GPU_EXCLUDE` masking applied in package `__init__`, `ops`, `llama` backend check), `training/accel_plan.py`
  (precision/optimizer/4-bit from the accelerator), `models/hf_loader.py` (GPU → 4-bit → `device_map=auto` OOM
  ladder), `fts accel` CLI. Wired into `training/{engine,worker,abliteration,vram/*}`, `testing/inference.py`,
  `models/{llama_loader,providers}`. Tests: `test_accel.py`, `test_accel_wiring_c1.py`.
- **Lane C2 — WebUI/RAG.** `webui/gpu_probe.py` (vendor-neutral memory probes) used by `routes/{system,models,pages}`,
  `_resources.html`, settings page; `data/rag_portable/devices.py` (`auto` = GPU first) in embedders/rerankers/
  store/standalone server (inline copy) / MCP package. Tests: `test_accel_wiring_c2.py`, UI/RAG test updates.
- Three lane commits are integrated: installer planner (B), training/loading/export (C1), WebUI/RAG (C2). Accelerator files are tracked.
- Final full suite ran on `8c54955` (VERSION .151). The concurrent docs-only commit `4ea403a` ran its version-bumping hook at 14:21 and changed VERSION to .152 during the suite, causing the two `tests/test_version.py` mismatches. With the tree settled, all 4 version tests pass. The GTX 1070 remains masked.

## Next steps

1. Final checks complete: `.venv/bin/ruff check src/ scripts/`, `make codemap-check`, and `git diff --check` pass; `tests/test_version.py` passes (4 tests).
2. Commit this final HANDOFF update; leave the branch unpushed until Genor explicitly approves.
3. Real-hardware check: `bash install.sh --plan`, `.venv/bin/fts accel`, then short LoRA train + GGUF export with `CUDA_VISIBLE_DEVICES=0`.
4. With Genor's OK: push, then run `bash update.sh` on fan-dragon and verify (`RESTART.md` cgroup check).
5. Dataset health follow-up: move `fts analyze` + `/api/data/analyze` onto `data/dataset_health.py`; retire `/api/data/hallucination-check`.
6. Wire `system_prompt` callers to the shared formatter; continue test-file review + tests-scope Ruff.

## Known issues

- No checkpoint resume in the training engine.
- Many endpoints still return 200 `{error}` instead of a 4xx/5xx.
- CLI has no project-dir fence (design call pending).
- `rag/sources` lists raw + parsed copies (possible double index); RAG coverage claims are filename-based, not content-hash.
- Holdout disjointness is exact-question only; reworded duplicates can leak.
- `compare` vs `suite` scores differ (judge path); `compare --models` swallows a trailing positional.
- Main DB is cwd-relative (`data/finetune_studio.db`); tests can leak empty dirs into `~/.finetune-studio/projects` and `test-fixtures/` (3 dirs from 2026-10-01 pre-date this work).
- `install.sh --repair` recreates the venv. Windows has no `run.ps1`/`update.ps1`.
- `training/data_quality.generate_fixes` suggests nonexistent CLI commands (`tests/test_data_quality_fixes.py`).
- `POST /api/data/upload` has no UI caller; `docs/modules/` still describes removed routes.
- External `TRANSFORMERS_CACHE` env noise (env-driven, not app code).
- Never exercised: chat-with-RAG E2E, QA mining via LM Studio, full dark/light visual pass, `rag.config.json`-only launch, GPU failure mid-train, AMD/Intel/Apple real hardware.

## Commands

- Tests: `.venv/bin/python -m pytest -q -p no:cacheprovider --ignore=tests/test_vram.py` (~21 min)
- Lane tests: `.venv/bin/python -m pytest -q tests/test_accel.py tests/test_accel_plan.py tests/test_accel_wiring_c1.py tests/test_accel_wiring_c2.py tests/test_install_diagnose_gpu.py tests/test_installer_contract.py`
- Ruff: `.venv/bin/ruff check src/ scripts/`
- GPU plan / health: `bash install.sh --plan`, `.venv/bin/fts accel`
- Codemap: `make codemap`

## Blockers

- Push and fan-dragon deploy wait for Genor's OK (and for the lanes to land).
