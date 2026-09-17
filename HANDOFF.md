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
| Evaluation contract | UI distinguishes held-out quality from memorization; training-eval responses retain raw transcript, judge input, scoring method, validity, errors, and provenance |
| Fidelity audit | `/api/projects/{pid}/data-prep/audit` checks raw hash, deterministic reparse, chunks, token coverage, pair provenance, chunk coverage, and export count. Benchmark audit independently recomputes every verdict. |
| Quality-v2 evidence | 230-row audited dataset; 156 optimizer steps / 6 epochs; merged + Q8 exported. Fresh held-out API run: 23/23 judged, 1 pass / 7 partial / 15 fail (4.3%). |
| Evidence audit | 230/230 leakage rows map to dataset questions; independent verdict recomputation matched 230/230 before stricter entity/contradiction rules. Parser audit is 35/35. |
| Last code | `fa84208` moves blocking Agent inference off the WebUI event loop; `9ea259f` adds provenance evidence and stricter source scoring. |

## Next steps
1. Deploy scorer/evidence changes and rerun held-out through WebUI; download raw transcript and call the benchmark audit endpoint.
2. Do not call parser fidelity “semantic completeness”: reconcile source fact inventory against approved pairs, especially tabular IDs and contact names.
3. Compare corrected held-out quality against full-dataset memorization; never report leakage as generalization: `.venv/bin/python -m pytest tests/test_training_eval.py tests/test_strict_scoring.py tests/test_fidelity_audit.py -q`.
4. If held-out remains poor, generate an augmented source-grounded dataset and retrain; preserve both raw runs before cleanup.
5. Re-run the browser smoke walkthrough after UI changes: `tests/run_qa.sh` (GPU/browser host only).
6. Verify retained model discovery after deployment: `curl -sSL http://fan-dragon:7860/api/models`.

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
- The current 230-pair dataset has chunk coverage but not semantic completeness: deterministic fact scan found uncovered IDs in `14_dispatch_metrics.csv` / `16_returns.csv`, `C-17` in `24_risk_register.csv`, and `INC-1842` in `03_incident_postmortem.md`.
- Local 27B Agent requests can exceed 180 seconds even at 2,048 tokens; the event-loop fix keeps health endpoints responsive, but the request must be quarantined rather than counted as generated data when it times out.
