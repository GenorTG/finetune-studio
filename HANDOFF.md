# Handoff — finetune-studio

## Mission

Local fine-tune, data-prep, and RAG WebUI. Current focus: context-grounded dataset rows and validating the end-to-end RAG effect.

## State (verified 2026-10-05)

| Area | State |
|---|---|
| Git | `main` and `origin/main` at `e6a2968`; clean before this handoff update. |
| Service | genorbox1 :7860 active; API reports 0.1.0.155 at `e6a29688`. RTX 3090 is the intended GPU; GTX 1070 remains excluded. |
| Grounded rows | `grounded_share` (default 40% when a RAG corpus exists) and `distractors` (0–2) are wired into export/subset and the data-prep UI. Shared RAG prompt is in `data/rag_portable/prompt.py`. |
| Live evidence | Qwen3-0.6B: one 150-step comparison, RAG grounding 2/5 plain → 4/5 grounded; unseen doc details 0/3 → 2/3. Single run/small question set; live distractor effect unmeasured. |
| Tests | Full suite on `e6a2968`: 1732 passed, 1 skipped, 0 failed, 1 deprecation warning; 21m32s. Ruff, codemap-check, diff-check pass. |
| Deploy | No fan-dragon deploy. Its checkout was previously reported to have a stub `.git`; restoring/recloning it was left undecided. |

## In flight

- Code change is pushed. Grounded-row selection is deterministic and order-independent; rows with stored chunks use their own source chunk and filename citation.
- Live RAG evidence indicates improvement, not proof of generalization. Distractor option has unit coverage, but no live RAG comparison with distractors.

## Next steps

1. Repeat live grounded-vs-plain evaluation on multiple seeds and a larger held-out question set; record per-question outcomes.
2. Measure live distractor rows against no-distractor rows, keeping training/evaluation conditions fixed.
3. Reconcile fan-dragon checkout (currently reported invalid) before any deploy; then use the documented `update.sh` route and verify commit/service/API.
4. If changing code: `.venv/bin/python -m pytest -q -p no:cacheprovider --ignore=tests/test_vram.py`, `.venv/bin/ruff check src/ scripts/`, `make codemap-check`.

## Known issues

- No `fts` command builds datasets; grounded options are currently exposed through the WebUI/API.
- RAG grounding comparison is n=1 with five questions; one unseen heater-power fact was answered with a memorized capacity value.
- Broader product follow-ups (API error semantics, CLI path fencing, installer/platform coverage, and RAG coverage) remain outside this grounded-row change.

## Commands

- Full tests: `.venv/bin/python -m pytest -q -p no:cacheprovider --ignore=tests/test_vram.py`
- Lint: `.venv/bin/ruff check src/ scripts/`
- Codemap: `make codemap-check`
- GPU plan/health: `bash install.sh --plan`, `.venv/bin/fts accel`

## Blockers

- Fan-dragon deploy requires resolving its invalid checkout; no deployment or host changes made.
