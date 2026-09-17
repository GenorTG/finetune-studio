# HANDOFF — finetune-studio

## Mission
Local fine-tune + data-prep WebUI (FastAPI, `src/finetune_studio/`).
Edit on genorbox1 → push → fan-dragon runs `finetune-studio.service` on :7860.

## State (verified 2026-09-17 Europe/Warsaw)
| Area | Status |
|------|--------|
| Release | EARLY BETA v0.1.0; service active under `finetune-studio.service` |
| Clean slate | `/api/projects=[]`, `/api/activity.tasks=[]`, inference unloaded |
| Retained models | Exactly Qwen3.8-27B GGUF + mmproj and Qwen3-4B Transformers |
| Prior realistic corpus | 43 human-readable originals uploaded/parsed; prior artifacts were intentionally cleaned after evaluation |
| Prior training/evaluation | 4B LoRA completed 116/116; old 47.7% score was same-data leakage, not held-out quality evidence |
| Evaluation contract | UI distinguishes held-out quality from memorization; benchmark cases now retain raw transcript, judge input, scoring method, validity, errors, and provenance |
| Fidelity audit | `/api/projects/{pid}/data-prep/audit` checks raw hash, deterministic reparse, chunks, token coverage, pair provenance, chunk coverage, and export count. Benchmark audit independently recomputes every verdict. |
| Tests | Full suite `849 passed, 3 warnings`; fidelity/scoring/API focus `52 passed, 2 warnings`; changed files Ruff-clean. Full-repo Ruff still has pre-existing unrelated findings. |
| Last code | Pending commit — deterministic parser/dataset/suite/raw-transcript audit and citation-aware scorer fix |

## Next steps
1. Upload a fresh realistic corpus and require `/api/projects/{pid}/data-prep/audit` to pass before curation: `curl -sSL http://fan-dragon:7860/api/projects/{pid}/data-prep/audit`.
2. Audit generated suite and raw evaluation before trusting a score: `curl -sSL http://fan-dragon:7860/api/projects/{pid}/benchmarks/{bid}/audit`.
3. Compare held-out quality against full-dataset memorization; never report the latter as generalization: `.venv/bin/python -m pytest tests/test_training_eval.py tests/test_strict_scoring.py tests/test_fidelity_audit.py -q`.
4. Re-run the browser smoke walkthrough after UI changes: `tests/run_qa.sh` (GPU/browser host only).
5. Verify retained model discovery after deployment: `curl -sSL http://fan-dragon:7860/api/models`.

## Commands
```
.venv/bin/python -m pytest tests/ -q --tb=short
.venv/bin/ruff check src/finetune_studio/data/audit.py src/finetune_studio/testing/audit.py
git push origin main
ssh fan-dragon "bash -lc 'cd /home/genortg/finetune-studio && git fetch origin main && git checkout --detach origin/main && systemctl --user restart finetune-studio'"
```

## Blockers
- The prior `256 → 230` count was the intended deterministic 90/10 train/validation split; new runs must report both counts.
- Training JSONL removes display-only source citations from assistant targets while retaining `source_id/chunk_idx` metadata.
- Numeric scorer provenance stripping must happen before both task detection and expected-value parsing; otherwise cited numeric answers are falsely classified as non-numeric.
- Fidelity is a chain, not a headline: raw bytes are reparsed and hashed, persisted chunks are compared with deterministic chunking, every pair maps to a source/chunk, suites report invalid/truncated rows, and benchmark audits expose every transcript plus independent verdict recomputation.
