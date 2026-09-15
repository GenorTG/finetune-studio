# HANDOFF — finetune-studio

## Mission
Local fine-tune + data-prep WebUI (FastAPI, `src/finetune_studio/`).
Edit on genorbox1 → push → fan-dragon pulls and runs `systemctl --user` unit `finetune-studio` on :7860.

## State (verified 2026-09-15 ~14:55 genorbox1)
| Area | Status |
|------|--------|
| Git | Uncommitted: SFT Hub-token KeyError fix + prior WebUI defaults (no commit per request) |
| push_to_hub_token | ✅ Fixed: build `SFTConfig` via `training/sft_args.py` (TRL 0.24 no longer pops missing key) |
| merge_on_save | ✅ Omitted → false |
| unsloth omitted | ✅ Omitted → false (standard TRL) |
| Tests | ✅ 13 passed: `test_sft_args.py` + `test_training_start_defaults.py` |
| Ruff | ✅ F821/F401 clean on touched training modules + new test |

## Next steps
1. Commit + push when Genor asks (include `sft_args.py`, engine/unsloth/profile wiring, `test_sft_args.py`).
2. On fan-dragon: pull + restart; confirm cgroup.
   `ssh fan-dragon 'bash -lc "cd ~/finetune-studio && git pull --ff-only && systemctl --user restart finetune-studio"'`
3. Browser: Qwen3-4B project Training → start (no Hub token) — must pass SFTTrainer init (no KeyError in ~8s).
4. Confirm Live status is not Unsloth; merge unchecked; step log visible.
5. Optional: Unsloth preset path still trains after SFTConfig switch.

## Commands
- Focused: `.venv/bin/python -m pytest tests/test_sft_args.py tests/test_training_start_defaults.py -v`
- Lint: `.venv/bin/python -m ruff check src/finetune_studio/training/sft_args.py src/finetune_studio/training/engine.py src/finetune_studio/training/unsloth_engine.py src/finetune_studio/training/vram/profile.py tests/test_sft_args.py --select F821,F401`
- Deploy: push then fan-dragon pull + `systemctl --user restart finetune-studio`

## Blockers
None for code. Live fix not on fan-dragon until commit/push + restart.
