# HANDOFF — finetune-studio

## Mission
Local fine-tune + data-prep WebUI (FastAPI, `src/finetune_studio/`).
Edit on genorbox1 → push → fan-dragon runs `finetune-studio.service` on :7860.

## State (verified 2026-09-15 Europe/Warsaw)
| Area | Status |
|------|--------|
| Abliterated export 500 | **Fixed (uncommitted)**: sync export returns JSON-safe `ExportResult`; numpy `refusal_direction` no longer hits `jsonable_encoder` |
| Export registry | Sync success for merged/abliterated/gptq/gguf creates a `model_exports` row (`done` + path/size) |
| Export UI | Readable non-JSON / `detail` errors; shows `output_path` / size / export_id |
| GGUF | Still fails truthfully (400 + converter/artifact message) when tools missing |
| Tests | `tests/test_export_response.py` + related export suites: 51 passed; Ruff clean on touched Python |

## Next steps
1. Commit export fix files (list below), then `git push`.
2. Fan-dragon: `cd /home/genortg/finetune-studio && git pull --ff-only && systemctl --user restart finetune-studio`.
3. Browser: Export → format=abliterated on a merged run → expect HTTP 200, path card, reload lists artifact (not HTTP 500).
4. Confirm GGUF without llama.cpp still shows readable 400 / converter message.
5. Truth check: `ss -ltnp \| grep 7860` pid cgroup contains `finetune-studio.service`.

## Commands
- Tests: `.venv/bin/python -m pytest tests/test_export_response.py tests/test_run_export.py tests/test_readable_results.py tests/test_export.py -v --tb=short`
- Lint: `.venv/bin/python -m ruff check src/finetune_studio/training/export_response.py src/finetune_studio/training/run_export.py src/finetune_studio/webui/routes/exports.py tests/test_export_response.py tests/test_readable_results.py`
- Deploy: `git push`; fan-dragon `git pull --ff-only && systemctl --user restart finetune-studio`

## Blockers
- GGUF conversion still needs llama.cpp tooling on the GPU host.
- Live fan-dragon still has prior incomplete HF trees under reset-backup — ops restore, not this checkout.
