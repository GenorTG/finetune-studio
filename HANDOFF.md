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
| Training | 4B LoRA run completed 116/116 steps; 27B global inference unloaded before training; merged safetensors produced |
| Q8 export | `model-q8_0.gguf` produced and loaded through Testing UI; export Open-in-Inference path now points to the actual file |
| Source suite | Auto-suite generated from approved JSONL: 256 cases, 0 skipped; Q8 result 122 pass / 85 partial / 49 fail = 47.7% |
| Generic Q8 smoke | Synthetic MMLU-shaped smoke: 6/6 = 100% against the actual Q8 file |
| Generic benchmark | Synthetic GSM8K-shaped smoke persisted: 1/6 = 16.7%; strict numeric judge correctly rejects provenance text after “final number” |
| Tests | Full suite `841 passed, 3 warnings`; targeted UI/model tests `20 passed, 2 warnings`; changed-route Ruff clean |
| Last code | `c0ad222` — exported GGUF paths and Open-in-Inference handlers |

## Next steps
1. If changing evaluation policy, decide whether strict numeric suites should strip approved provenance citations: `.venv/bin/python -m pytest tests/test_training_eval.py -q`.
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
- The approved JSONL contains 256 rows, while the trainer reported 230 usable examples; this should be reconciled before treating dataset accounting as production-grade.
- The 47.7% source-suite score is an honest result, not a quality claim; partial/failing cases need review before a serious adapter release.
- Generic smoke output includes source citations, which is desirable for provenance but conflicts with “final number only” strict numeric prompts.
