# HANDOFF — finetune-studio

## Mission
Local fine-tune + data-prep WebUI (FastAPI, `src/finetune_studio/`).
Edit on genorbox1 → push → fan-dragon runs `finetune-studio.service` on :7860.

## State (verified 2026-09-15 16:36 Europe/Warsaw)
| Area | Status |
|------|--------|
| Git | `5e634a8` deployed; service active under `finetune-studio.service` |
| UI updates | SSE live events replace 2s full-panel redraws; browser saw `200 eventsource` |
| Results UI | Main testing/export result surfaces use readable grids/cards; raw JSON is debug-only |
| Export formats | AWQ removed; unsupported GPTQ fails honestly; supported choices are not over-advertised |
| Regression | 50 focused tests passed; Ruff passed on touched Python files |
| Runtime reset | Fan-dragon project DB, provider DBs, project data, and generated output reset to zero |
| Fresh browser state | Dashboard shows 0 projects; exactly one provider: retained 27B Qwen GGUF helper |
| Retained models | Gemma 4B safetensors for training; Qwen3.8-27B Q4_K_M GGUF + projector for inference help |
| Reset backup | `/home/genortg/finetune-studio-reset-backups/20260915-163033` on fan-dragon |
| Repo hygiene | DB, SQLite, runtime data, models, output, projects, and private artifacts are gitignored |

## Next steps
1. Create a new project in the browser: open `http://fan-dragon:7860/` and click `CREATE YOUR FIRST PROJECT`.
2. Verify data-prep SSE: use the project Data Prep page and confirm the live activity stream stays connected at `/api/activity/events`.
3. Run a fresh 4B training flow only after selecting the prepared dataset in the browser; verify live logs and final status in the UI.
4. Add a real GGUF converter before exposing GGUF export as successful; never trust an HTTP 200 without an artifact row.
5. Add industry suites and a 27B-assisted project suite through browser-visible workflows before benchmarking.

## Commands
- Tests: `.venv/bin/python -m pytest tests/ -v --tb=short`
- Lint: `.venv/bin/python -m ruff check src/`
- Deploy: `git push`; fan-dragon `git pull --ff-only && systemctl --user restart finetune-studio`
- Service truth: `systemctl --user show -p MainPID --value finetune-studio` then `grep finetune-studio.service /proc/<pid>/cgroup`

## Blockers
- No project/data/training artifacts remain after the authorized fan-dragon reset.
- GGUF conversion and GPTQ require backend work; AWQ is intentionally not advertised.
- Industry-standard suites and fresh 27B-generated test-suite preparation remain to be implemented.
