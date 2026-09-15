# HANDOFF — finetune-studio

## Mission
Local fine-tune + data-prep WebUI (FastAPI, `src/finetune_studio/`).
Edit on genorbox1 → push → fan-dragon runs `finetune-studio.service` on :7860.

## State (verified 2026-09-15 16:50 Europe/Warsaw)
| Area | Status |
|------|--------|
| Git | Industry smoke + GGUF reliability + discovery test updates uncommitted (no push yet) |
| Industry smoke | Built-in `mmlu_smoke` / `gsm8k_smoke` / `hellaswag_smoke` via `suite_defs` + `fixtures/` |
| Discovery tests | Updated for always-on builtins; local + auto coverage kept |
| Project testing | Dropdown asserts industry labels + discover path/label match |
| Regression | 17 focused discovery/testing/smoke tests passed; Ruff clean on touched files |
| Fan-dragon | Still at prior deploy until push + pull + restart |

## Next steps
1. Review + commit when ready: `git status` / `git diff tests/test_benchmarks_suite_discovery.py tests/test_project_testing.py`.
2. Include related untracked `suite_defs.py`, `fixtures/`, `tests/test_benchmarks_industry_smoke.py` in the same commit if shipping industry smoke.
3. Push and on fan-dragon: `cd /home/genortg/finetune-studio && git pull --ff-only && systemctl --user restart finetune-studio`.
4. Browser: `/projects/{pid}/benchmarks` and `/testing` — suite select shows `industry · … (smoke v1)`.
5. Create a new project after the prior reset: `http://fan-dragon:7860/` → CREATE YOUR FIRST PROJECT.
6. Install llama.cpp converter on fan-dragon before treating GGUF export as available.

## Commands
- Tests: `.venv/bin/python -m pytest tests/test_benchmarks_suite_discovery.py tests/test_project_testing.py tests/test_benchmarks_industry_smoke.py -v --tb=short`
- Lint: `.venv/bin/python -m ruff check tests/test_benchmarks_suite_discovery.py tests/test_project_testing.py tests/test_benchmarks_industry_smoke.py src/finetune_studio/benchmarks/suite_defs.py`
- Deploy: `git push`; fan-dragon `git pull --ff-only && systemctl --user restart finetune-studio`
- Service truth: `systemctl --user show -p MainPID --value finetune-studio` then `grep finetune-studio.service /proc/<pid>/cgroup`

## Blockers
- GGUF conversion still needs llama.cpp tooling on the GPU host; API fails honestly until present.
- GPTQ still needs `auto_gptq` where used; AWQ intentionally not advertised.
- Changes not committed/pushed — fan-dragon cannot see industry smoke until deploy.
