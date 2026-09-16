# HANDOFF — finetune-studio

## Mission
Local fine-tune + data-prep WebUI (FastAPI, `src/finetune_studio/`).
Edit on genorbox1 → push → fan-dragon runs `finetune-studio.service` on :7860.

## State (verified 2026-09-16 Europe/Warsaw)
| Area | Status |
|------|--------|
| Synthetic benches | Suite types `synthetic_smoke` / `synthetic_offline`; labels `synthetic · … (not industry)`; fixtures flag `is_industry_benchmark: false` |
| Scoring | MCQ/numeric use `strict_scoring` (exact option / normalized final); records `scoring_method` + `validity`; keyword substring only for open-ended |
| Dashboard hero | Fake CPU/MEM/nodes/last-scan removed; `{% include "_resources.html" %}` polls `/api/system/resources` |
| Tests | Focused reliability suite green locally; changed-file Ruff clean |

## Next steps
1. `git add` + commit reliability pass when Genor asks; then `git push` and fan-dragon `git pull --ff-only && systemctl --user restart finetune-studio`.
2. Browser-check dashboard hero RAM/VRAM bars after deploy (`curl -s http://127.0.0.1:7860/api/system/resources`).
3. Keep reporting synthetic suite scores as local synthetic — never as industry MMLU/GSM8K/HellaSwag.

## Commands
- Reliability tests: `.venv/bin/python -m pytest tests/test_strict_scoring.py tests/test_dashboard_template.py tests/test_benchmarks_offline_suites.py tests/test_benchmarks_industry_smoke.py tests/test_benchmarks_suite_discovery.py tests/test_benchmarks_template.py tests/test_project_testing.py tests/test_status_badge_honesty.py tests/test_bench_judge.py -v --tb=short`
- Lint: `.venv/bin/ruff check src/finetune_studio/testing/strict_scoring.py src/finetune_studio/testing/suite.py src/finetune_studio/benchmarks/suite_defs.py src/finetune_studio/benchmarks/offline_suites.py src/finetune_studio/webui/routes/benchmarks.py src/finetune_studio/webui/routes/testing.py tests/test_strict_scoring.py tests/test_dashboard_template.py`
- Deploy: `git push`; fan-dragon `git pull --ff-only && systemctl --user restart finetune-studio`

## Blockers
- Changes are local/uncommitted until Genor requests commit + push.
