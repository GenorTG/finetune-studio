# HANDOFF — finetune-studio

## Mission
Local fine-tune + data-prep WebUI (FastAPI, `src/finetune_studio/`).
Edit on genorbox1 → push → fan-dragon runs `finetune-studio.service` on :7860.

## State (verified 2026-09-15 Europe/Warsaw)
| Area | Status |
|------|--------|
| Benchmark NO_DATA | **Fixed**: compare/cases empties strip `.empty` chrome via `hideCmpEmpty` + CSS `.empty[hidden]` / `.is-filled`; cases empty omitted when results exist |
| Benchmark RUN | Accepts status `done`/`completed`; disabled runs show why |
| Testing Run | Starts disabled until suite picked; clear status copy |
| Export converters | `export_capabilities` probes GGUF/GPTQ; unavailable formats disabled + actionable hints; merged/abliterated remain |
| Raw JSON | Primary result panels stay tables/kv grids (Debug JSON only under `<details>`) |
| Tests | 43 focused passed; Ruff clean on touched Python |

## Next steps
1. Commit + `git push` (browser-result surfaces + export capabilities).
2. Fan-dragon: `cd /home/genortg/finetune-studio && git pull --ff-only && systemctl --user restart finetune-studio`.
3. Browser: Benchmarks with results — no `┌── NO_DATA ──┐` beside case/score tables; Compare after load hides empty chrome.
4. Browser: Export — GGUF/GPTQ show unavailable when converters missing; Export selected refuses disabled formats.
5. Browser: Testing — Run disabled until suite picked; status line updates.
6. Truth check: `ss -ltnp \| grep 7860` pid cgroup contains `finetune-studio.service`.

## Commands
- Tests: `.venv/bin/python -m pytest tests/test_export_capabilities.py tests/test_browser_result_surfaces.py tests/test_testing_run_gate.py tests/test_benchmarks_compare.py tests/test_benchmarks_template.py tests/test_readable_results.py tests/test_awq_removed.py tests/test_export_response.py -v --tb=short`
- Lint: `.venv/bin/python -m ruff check src/finetune_studio/training/export_capabilities.py src/finetune_studio/webui/routes/pages.py tests/test_export_capabilities.py tests/test_browser_result_surfaces.py tests/test_testing_run_gate.py tests/test_benchmarks_compare.py tests/test_readable_results.py`
- Deploy: `git push`; fan-dragon `git pull --ff-only && systemctl --user restart finetune-studio`

## Blockers
- GGUF still needs llama.cpp (`convert_hf_to_gguf.py`) on the GPU host.
- GPTQ still needs `auto_gptq` where used.
- Current browser-result changes must be deployed before the final visual pass.
