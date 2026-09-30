# HANDOFF — finetune-studio

## Mission
Trained models must reliably answer the learned corpus without inventing database sources. Every parsed chunk must reach the training dataset; judge by transcript review, not auto-green tests.

## State (verified 2026-09-30)
- GPTQ export support was removed by decision; GGUF is the local-inference path.
- The per-file data-prep pipeline is hardened and was proven on the stealth-dragon GPU.
- Fan-dragon Gemma 4 12B completed source-build → load → chat round-trip verification. Fan-dragon Qwen3-30B-A3B and both genorbox1 helper smoke-loads remain pending.
- genorbox1 has about 1.1 TB free; do not pull additional models without Genor's approval.
- Driver 535 caps torch at cu124; newer torch requires driver ≥580. Visual changes require rendered-page checks, not HTTP status alone.
- **Full app proven end-to-end at real scale on genorbox1 (RTX 3090)**, project `564cd0c2` "Large Corpus Stress Test": 102 files across all 12 supported source formats (txt/md/html/csv/json/jsonl/xml/docx/xlsx/pptx/pdf/rtf) uploaded and parsed in one batch, 0 failures. QA generation mined 515 pairs across all 102 sources (Gemma 4 12B helper, full GPU offload). Dataset build: 199 training rows, 100% coverage. Real training run (Qwen3-0.6B, smoke preset) completed with genuine decreasing loss (3.94 → 0.69). GGUF export chain completed: merge (1.12 GB) → f16 GGUF → q4_k_m (0.37 GB) → q5_k_m, all real artifacts with correct sizes. Every step driven through the actual browser UI, not the API directly.
- Fixed a real bug found during this pass: `fd3b843` — the project wizard's Step 4/5 status widgets and its auto-test flow called `GET /api/training/runs?project_id=PID`, but that query string is silently ignored by the `/runs` route (which lists every project's runs); the project-scoped route is `/runs/{pid}`. A brand-new project's wizard was showing another project's run count and loss. Fixed all three call sites.

## Known issues (found 2026-09-30, not yet fixed)
1. **RAG document count inconsistency at scale**: uploading 102 files and building the RAG index reported "Done — 120 docs, 120 chunks indexed", but the actual Documents table showed 127 rows. No duplicate filenames were found in the visible rows, so root cause is unconfirmed — could be a stale/partial index from an earlier build not being cleared, or a display vs. persisted-count mismatch. Needs a from-scratch rebuild + count audit to isolate.
2. **New-project base-model dropdown shows duplicate entries**: 4 indistinguishable rows all labeled "Aethermoor World Bible · Qwen3.5-9B · merged (safetensors, 16.68 GB)" — likely 4 different export runs sharing a display label with no way to tell them apart (missing run ID or timestamp in the label).
3. **No image support in the parser registry** (`data/parsers.py`): `.png/.jpg/.pdf-images` etc. are not a supported source type at all — Pillow is a dependency but not wired to any content-extraction path. Worth a product decision (intentional scope vs. gap) rather than assuming it's a bug.

## Next steps
1. Polish Versions UX: compare manifests side-by-side and support copy-pins-to-new-project in `webui/routes/versions.py` and wizard step 6.
2. Smoke-load the remaining helpers through `POST /api/models/load` on both hosts; record VRAM/round-trip results.
3. Run the full-coverage benchmark protocol in `docs/judging/PROTOCOL.md` before comparing verdict percentages.
4. Keep visual changes gated by real rendered-page sweeps and saved screenshots.
5. Investigate the RAG doc-count and base-model-dropdown-label issues above.

## Commands
- Install/check: `bash install.sh && bash install.sh --check`.
- Update: `bash update.sh`.
- Restart either host service: `systemctl --user restart finetune-studio`.
- Inspect logs: `journalctl --user -u finetune-studio -n 100 --no-pager`.

## Blockers
- Fan-dragon Qwen3-30B-A3B is tight on the 16 GB card; IQ4_XS, q4_0 KV, or partial CPU offload may be required.
- Remaining helper smoke-loads and visual benchmark evidence are pending.
