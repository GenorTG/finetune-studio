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
| Realistic corpus | 43 human-readable originals (prose, HTML, CSV, JSONL, PDF, DOCX, XLSX) uploaded and parsed with 0 errors; originals/provenance verified in WebUI |
| Curation | 256 approved source-grounded Q&A rows; 0 duplicate question/source groups; every answer cited its source filename |
| Training | 4B LoRA run completed 116/116 steps; 27B global inference unloaded before training; merged safetensors produced. Training now reports train/validation counts explicitly and uses a deterministic split. |
| Q8 export | `model-q8_0.gguf` produced and loaded through Testing UI; export Open-in-Inference path now points to the actual file |
| Source suite | Auto-suite generated from approved JSONL: 256 cases, 0 skipped; the old Q8 result was 122 pass / 85 partial / 49 fail = 47.7%, a same-data leakage check rather than held-out quality evidence |
| Generic Q8 smoke | Synthetic MMLU-shaped smoke: 6/6 = 100% against the actual Q8 file |
| Generic benchmark | Synthetic GSM8K-shaped smoke persisted: 1/6 = 16.7%; citation filenames polluted numeric extraction. Provenance is now stripped from training targets and strict numeric scoring. |
| Evaluation | Testing UI now distinguishes deterministic held-out validation (quality) from full training-set memorization/leakage evaluation. |
| Tests | Full suite `844 passed, 3 warnings`; focused remediation `37 passed, 2 warnings`; changed non-training files Ruff-clean. Full-repo Ruff still has pre-existing unrelated findings. |
| Last code | `07d6239` — clean training targets and held-out evaluation |

## Next steps
1. Re-run a fresh realistic corpus through Data Prep, then compare held-out vs leakage scores; do not call the leakage score generalization: `.venv/bin/python -m pytest tests/test_training_eval.py tests/test_strict_scoring.py -q`.
2. Re-run the browser smoke walkthrough after UI changes: `tests/run_qa.sh` (GPU/browser host only).
3. Verify retained model discovery after deployment: `curl -sSL http://fan-dragon:7860/api/models`.

## Commands
```
.venv/bin/python -m pytest tests/ -q --tb=short
.venv/bin/python -m ruff check src/finetune_studio/webui/routes/pages.py src/finetune_studio/webui/routes/training.py
git push origin main
ssh fan-dragon "bash -lc 'cd /home/genortg/finetune-studio && git fetch origin main && git checkout --detach origin/main && systemctl --user restart finetune-studio'"
```

## Blockers
- The prior `256 → 230` count was the intended deterministic 90/10 train/validation split; the UI previously omitted the held-out count. New runs report both.
- The prior 47.7% score mixed same-data leakage evaluation with heuristic open-ended judging; use the new held-out mode for quality claims.
- Training JSONL now removes display-only source citations from assistant targets while retaining source_id/chunk_idx metadata; this prevents filename digits from contaminating answers and scores.
