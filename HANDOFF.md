# HANDOFF — finetune-studio

## Mission
Local fine-tune + data-prep WebUI (FastAPI, `src/finetune_studio/`).
Edit on genorbox1 → push → fan-dragon runs `finetune-studio.service` on :7860.

## State (verified 2026-09-15 ~15:45 Europe/Warsaw)
| Area | Status |
|------|--------|
| Git | output_path persist fix ready to push (training start → DB) |
| Training start | ✅ After scoping `output_dir`, `db.update_run(..., output_path=…)` runs immediately |
| Regression | ✅ `tests/test_training_start_defaults.py` asserts DB keeps scoped + explicit paths |
| Export API | ✅ Adapter-only merge-at-export still in place (prior `6a5e5f8`) |
| Run `38867d1b` | ⚠ Live row still has empty `output_path` until backfilled or re-trained under new code |

## Next steps
1. On fan-dragon: `cd /home/genortg/finetune-studio && git pull --ff-only && systemctl --user restart finetune-studio`
2. Backfill run `38867d1b` if artifacts exist on disk (typical path):  
   `curl -s -X POST http://127.0.0.1:7860/api/training/runs/38867d1b/set-output -H 'Content-Type: application/json' -d '{"output_path":"output/projects/04954e70/runs/38867d1b"}'`
3. Export that run as GGUF `q8_0` (+ optional GPTQ) with a 16-bit Qwen3-4B base:  
   `http://fan-dragon:7860/projects/04954e70/export?run=38867d1b`
4. Confirm Training detail shows Output path (not `—`) for a newly started run.
5. Run project Testing against each exported format.

## Commands
- Focused: `.venv/bin/python -m pytest tests/test_training_start_defaults.py -q --tb=short`
- Lint: `.venv/bin/python -m ruff check src/finetune_studio/webui/routes/training.py tests/test_training_start_defaults.py`
- Deploy: `git push`; on fan-dragon `git pull --ff-only && systemctl --user restart finetune-studio`
- Service truth check: `systemctl --user show -p MainPID --value finetune-studio; grep finetune-studio.service /proc/<pid>/cgroup`

## Blockers
- Existing completed runs created before this fix still need a one-shot `set-output` (or retrain) before Export can see them.
