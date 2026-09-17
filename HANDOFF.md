# HANDOFF — finetune-studio

Local fine-tune + data-prep WebUI (FastAPI, `src/finetune_studio/`).
Edit on genorbox1 → push → fan-dragon runs `finetune-studio.service` on :7860.

## Mission
Ship held-out quality + WebUI evidence that an outside reviewer can't dispute:
real root-cause fixes, real fan-dragon verification, no fake greens. Every
audit row is reproducible from raw artifacts in `.tmp/evidence-run/` and
the project's per-source qa/pairs/*.json.

## State (verified 2026-09-17 17:20 CEST)
| Area | Status |
|------|--------|
| Release | EARLY BETA v0.1.0; service active under `finetune-studio.service` |
| Retained project | `fbcf7083` (Helios Fulfillment Evidence Run); models Qwen3.8-27B GGUF helper + Qwen3-4B base; runs `8587cee6` / `8b1dd006` / `f4f627af` |
| Quality-v3 held-out | **4 pass / 7 partial / 12 fail = 32.6% weighted, 17.4% strict** (benchmark `ed1af82e`) |
| Source-disjoint v4 | **0 pass / 12 partial / 40 fail = 11.5% weighted** on 52 unseen-source cases (`bb896ad9`) — fine-tune alone does not generalize |
| **RAG-grounded suite** | `testing/rag_suite.py` + `POST /api/testing/run-rag-suite` landed locally (uncommitted) |
| Live UI | Fan-dragon `e2e_ui_qa.py` last green: **70/70** |
| Prior HANDOFF | `docs/archive/HANDOFF-2026-09-17-pre-rag-suite.md` |

### RAG-suite (this milestone)
- **Module:** `src/finetune_studio/testing/rag_suite.py` — reusable. Loads PortableRAG (`corpus_path` or default `~/.finetune-studio/rag_corpora/<pid>` / `project_rags`), top-k retrieve per `BenchmarkCase`, context-only prompt with `UNKNOWN_REPLY` fallback, preserves transcript/context/hits/provenance/`model_path`/`corpus_path`, applies existing heuristic scoring, reports retrieval hit/recall.
- **Endpoint:** `POST /api/testing/run-rag-suite` — `_ensure_model_loaded` + blocking eval via `asyncio.to_thread`. No UI changes.
- **Verify (genorbox1, this session):**
  - `.venv/bin/python -m pytest tests/test_rag_suite.py -v --tb=short` → **10 passed** in 3.86s
  - `.venv/bin/python -m ruff check src/finetune_studio/testing/rag_suite.py src/finetune_studio/webui/routes/testing.py tests/test_rag_suite.py` → **All checks passed!**

## Next steps
1. Parent review → commit + push when approved; fan-dragon `git pull --ff-only` + `systemctl --user restart finetune-studio`.
2. Fan-dragon live check: `POST /api/testing/run-rag-suite` on the 52-case source-disjoint suite + PortableRAG corpus for `fbcf7083` (expect retrieval recall ≫ fine-tune-only 0/52).
3. Optional: Testing-tab “Run with RAG” button (not in this milestone).
4. Keep browser QA green (70/70).

## Commands
```
.venv/bin/python -m pytest tests/test_rag_suite.py -v --tb=short
# → 10 passed

.venv/bin/python -m ruff check \
  src/finetune_studio/testing/rag_suite.py \
  src/finetune_studio/webui/routes/testing.py \
  tests/test_rag_suite.py
# → All checks passed!
```
- Full suite: `make test`
- Deploy (after commit): `git push` → fan-dragon pull + `systemctl --user restart finetune-studio`

## Blockers
- Milestone left **uncommitted** by request (no push).
- Browser upload MiniMax `paths` quirk unchanged; use API upload.
- Training OOM from stale GPU processes — check `nvidia-smi --query-compute-apps=pid,used_memory --format=csv` before runs.
