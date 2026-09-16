# End-to-end workflow execution plan

## Objective
Prove and harden the real user journey: upload heterogeneous source material, preserve and review parsed content, use the 27B inference model for data-prep assistance, train the 4B Transformers model, evaluate it with a source-scoped suite, export a merged Q8 GGUF, and benchmark the result.

## Execution sequence
1. **Baseline and clean state** — verify service ownership, no loaded inference model, empty projects/activity, and exactly the two retained models.
2. **Source corpus** — create at least 30 manually authored files across plain text, Markdown, CSV, JSON, JSONL, YAML, XML, HTML, INI, TOML, TSV, log, SQL, Python, JavaScript, TypeScript, shell, and document-like formats. Keep a manifest with source intent and expected facts.
3. **Ingest and review** — create one project, upload every source through the WebUI, wait for parsing, verify originals remain, inspect metadata provenance, preview raw/parsed/version/conversion views, edit and version a sample, and test select-all/file filtering.
4. **27B-assisted Q&A** — load the retained Qwen3.8-27B GGUF only for inference/data-prep assistance. Generate a comprehensive ShareGPT dataset covering every manifest fact, with traceability back to source and version. Confirm the UI explains model role, scope, progress, and failure states.
5. **4B training** — unload 27B automatically before training; configure a conservative LoRA run for Qwen3-4B, start it, monitor live status and Activity concurrently, and verify loss, steps, output, and cancellation/error handling.
6. **Evaluation suite** — create a large source-scoped test suite from the generated Q&A, run it against the trained adapter, inspect per-case answers/verdicts/reasoning and aggregate scores, and verify Activity links to the correct project/run.
7. **Export and benchmarks** — merge the adapter, export a clearly named Q8 GGUF, load it for generic and source-scoped benchmarks, verify benchmark rows are judged (not merely executed), and inspect comparison/history views.
8. **Fix and prove** — capture screenshots at desktop/mobile widths for each major state, fix every reproduced UI/copy/data/progress defect directly, rerun focused and full tests, then remove only disposable projects/artifacts while preserving the two model assets.

## Acceptance checks
- Originals, parsed files, edits, versions, source hashes, dataset provenance, and suite provenance remain inspectable.
- No action claims success before its backend result exists; unavailable actions are disabled or explain the prerequisite.
- 27B is never resident concurrently with training; Activity shows current and completed tasks with truthful status/project/run links.
- Export names identify model, adapter, quantization, and source dataset.
- Final runtime is clean: no temporary projects/activity, inference unloaded, exactly two retained model assets.
