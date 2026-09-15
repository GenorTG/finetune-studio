# HANDOFF — finetune-studio

## Mission
Local fine-tune + data-prep WebUI (FastAPI, `src/finetune_studio/`).
Edit on genorbox1 → push → fan-dragon runs `finetune-studio.service` on :7860.

## State (verified 2026-09-15 Europe/Warsaw)
| Area | Status |
|------|--------|
| Git | Industry smoke suite foundation on genorbox1 (uncommitted until asked) |
| Industry suites | Offline smoke fixtures: MMLU / GSM8K / HellaSwag-style (`benchmarks/fixtures/*.v1.json`) |
| Discovery | `suite_defs.discover_suites` → industry + local `data/benchmarks` + project auto |
| UI | Benchmarks catalog shows type/source/cases; Testing dropdown uses readable labels |
| Regression | 23 focused tests passed; Ruff clean on touched Python files |
| Prior deploy | `5e634a8` still the last fan-dragon deploy reference until push |

## Next steps
1. Review + commit the industry-suite foundation on genorbox1 when ready.
2. `git push`; fan-dragon: `cd /home/genortg/finetune-studio && git pull --ff-only && systemctl --user restart finetune-studio`.
3. Create a project in the browser and confirm Benchmarks lists the three industry smoke suites with labels.
4. Run a base-model or trained-run smoke suite and confirm the per-case table (no raw JSON).
5. Add a real GGUF converter before exposing GGUF export as successful.
6. Optional later: 27B-assisted project suite generation through browser workflows.

## Commands
- Focused tests: `.venv/bin/python -m pytest tests/test_benchmarks_industry_smoke.py tests/test_benchmarks_suite_discovery.py tests/test_benchmarks_template.py tests/test_project_testing.py -v --tb=short`
- Lint: `.venv/bin/python -m ruff check src/finetune_studio/benchmarks/suite_defs.py src/finetune_studio/webui/routes/benchmarks.py`
- Deploy: `git push`; fan-dragon `git pull --ff-only && systemctl --user restart finetune-studio`
- Service truth: `systemctl --user show -p MainPID --value finetune-studio` then `grep finetune-studio.service /proc/<pid>/cgroup`

## Blockers
- Changes not pushed / not restarted on fan-dragon yet.
- GGUF conversion and GPTQ still need backend work; AWQ intentionally not advertised.
- Full HF industry datasets remain CLI-only (`real_benchmarks.py`); browser uses local smoke fixtures only.
