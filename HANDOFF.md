# HANDOFF — finetune-studio

## Mission
Local fine-tune + data-prep WebUI (FastAPI, `src/finetune_studio/`).
Edit on genorbox1 → push → fan-dragon pulls and runs `systemctl --user` unit `finetune-studio` on :7860.

## State (verified 2026-09-15 ~14:30 genorbox1)
| Area | Status |
|------|--------|
| Git | Uncommitted Qwen3-4B WebUI defect fixes on `main` (no commit per request) |
| merge_on_save | ✅ Omitted → false; explicit true / FormData `"1"` preserved (`training.py`) |
| unsloth omitted | ✅ Omitted → false (standard TRL; no “Loading model with Unsloth…” on stock Qwen3-4B form path) |
| Training live log | ✅ Project + global training pages show step log (not `display:none` / dim) |
| Testing live status | ✅ Polls existing `/api/testing/status` + elapsed while suite runs |
| Tests | ✅ 16 passed: start defaults, monitor template, project testing |
| Ruff | ✅ `training.py` clean (removed dup `/runs` handlers; callback `# noqa: BLE001, S110`) |

## Next steps
1. Commit + push when Genor asks (stage only the 8 files from this fix if separating from other WIP).
   `git status` then selective `git add` of training/testing WebUI paths.
2. On fan-dragon: pull + `systemctl --user restart finetune-studio`; confirm cgroup.
   `ssh fan-dragon 'bash -lc "cd ~/finetune-studio && git pull --ff-only && systemctl --user restart finetune-studio"'`
3. Browser: project Training — start without merge checkbox; Live status must not say Unsloth; Step log visible.
4. Browser: Testing — Run suite; `t-status` elapsed + `t-live-log` model line update while waiting.
5. Optional: confirm presets that send `unsloth: true` in overrides still use Unsloth.

## Commands
- Focused tests: `.venv/bin/python -m pytest tests/test_training_start_defaults.py tests/test_training_monitor_template.py tests/test_project_testing.py -v`
- Lint: `.venv/bin/python -m ruff check src/finetune_studio/webui/routes/training.py --select F821,F401`
- Deploy: push then fan-dragon pull + `systemctl --user restart finetune-studio`

## Blockers
None for code. Suite runs are still synchronous — live UI only shows elapsed + inference load status, not per-case progress (no suite activity source in `/api/activity`).
