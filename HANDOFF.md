# HANDOFF — finetune-studio

## Mission
Local fine-tune + data-prep WebUI (FastAPI, `src/finetune_studio/`).
Edit on genorbox1 → push → fan-dragon runs `finetune-studio.service` on :7860.

## State (verified 2026-09-15 ~16:20 Europe/Warsaw)
| Area | Status |
|------|--------|
| Git | `dec5d10` deployed; service active under `finetune-studio.service` |
| 4B training | ✅ Browser run `598d9d14`: Qwen3-4B safetensors, raw LoRA, 172/172, live logs, DONE |
| Activity drawer | ✅ Expanded task survives the 2-second refresh; browser verified |
| Merge/export | ✅ Browser merge-at-export produced 7.5 GB safetensors; raw adapter remains separate |
| Testing auto-load | ✅ Accepts status `done`/`completed`; requires ready `merged/` weights |
| Testing model list | ✅ Project-scoped exports from `_scan_run_models` (not global HF list); latest merged pre-selected |
| Testing suite | ✅ Browser run: 10 judged, 8 passed, 1 partial, 1 failed (80%) with per-case table/logs |
| Project benchmark | ✅ Browser run: 10 judged, 8 passed, 1 partial, 1 failed (80%); results and scores rendered |
| Export failures | ✅ Sync path returns HTTP 400 + `{ok:false,status:failed,error}`; UI notifies + `role=status` |
| GPTQ missing dep | ✅ Fail-fast when `auto_gptq` absent (no fake success) |
| Regression | ✅ 46 focused tests green; ruff clean on touched files |
| Model cleanup | ✅ Old 0.6B/Unsloth/Gemma caches and artifacts removed; 4B/27B plus RAG embedder retained |

## Next steps
1. Browser: investigate GGUF `q8_0`; POST returned 200 but no artifact row appeared after refresh, so it is not verified.
2. Browser: confirm GPTQ failure is now shown as an error notification, not success; install `auto-gptq` only if a real GPTQ artifact is required.
3. Implement a real AWQ backend before advertising AWQ; current UI correctly omits/marks unsupported paths.
4. Add/import industry-standard suites (only project `default.json` exists currently) and generate a fresh 27B-assisted testing suite rather than reusing the existing fixture.
5. Service truth check: `ss -ltnp | grep 7860` → `/proc/<pid>/cgroup` contains `finetune-studio.service`.

## Commands
- Focused: `.venv/bin/python -m pytest tests/test_testing_auto_load.py tests/test_testing_models.py tests/test_project_testing.py tests/test_run_export.py tests/test_export.py -q --tb=short`
- Lint: `.venv/bin/python -m ruff check src/finetune_studio/webui/testing_models.py src/finetune_studio/webui/routes/testing.py src/finetune_studio/training/run_export.py`
- Deploy: `git push`; fan-dragon `git pull --ff-only && systemctl --user restart finetune-studio`

## Blockers
- GGUF conversion is still unproven: browser POST returned HTTP 200 but no artifact appeared.
- GPTQ needs `auto-gptq` on fan-dragon for a successful export.
- AWQ has no working backend in this environment.
- Industry-standard benchmark suites and a fresh 27B-generated test suite are not yet wired into this project flow.
