# HANDOFF — finetune-studio

## Mission
Scrub public-facing README + GitHub Pages (`docs/index.html`) of private infra
fingerprints and stale capability claims; lock with a docs scrub test.

## State (verified 2026-09-15 Europe/Warsaw)
| Area | Status |
|------|--------|
| README.md | Rewritten: no hostnames/paths/PIDs/ComfyUI; honest GGUF/GPTQ; offline smoke benches |
| docs/index.html | Same scrub + accurate export/benchmark/E2E wording |
| Linked public docs | Surgical hostname scrub in ARCHITECTURE / DEPENDENCIES / DEPLOYMENT / REFACTOR-SPEC |
| Scrub test | `tests/test_public_docs_scrub.py` — **20 passed**; ruff clean |
| Unrelated dirty tree | Leave alone: export UI/css/pages + `export_capabilities.py` (not this task) |

## Next steps
1. Review diff, then commit only the public-docs files + scrub test (ask Genor).
2. `git push` after commit so GitHub Pages picks up `docs/index.html`.
3. Optional: scrub `tests/README_E2E.md` (still names fan-dragon) — not linked from README.
4. Resume prior export/abliterated work from `docs/archive/HANDOFF-2026-09-15-pre-public-docs-scrub.md` if still needed.

## Commands
- Tests: `.venv/bin/python -m pytest tests/test_public_docs_scrub.py -v --tb=short`
- Lint: `.venv/bin/python -m ruff check tests/test_public_docs_scrub.py`
- Diff scope: `git diff -- README.md docs/index.html docs/ARCHITECTURE.md docs/DEPENDENCIES.md docs/DEPLOYMENT.md docs/REFACTOR-SPEC.md tests/test_public_docs_scrub.py`

## Blockers
- None for the docs scrub. Commit/push not requested yet.
