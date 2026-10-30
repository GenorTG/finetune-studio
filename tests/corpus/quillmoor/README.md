# Quillmoor benchmark corpus

A generated, fully known world for measuring what Finetune Studio does with real-looking company files: does every
generated format parse, does Smart Brain find the right passage, does an answer model use it, and how much does a fine-tuned
adapter remember on its own. Nothing here is real, so a model can only answer from what the app gave it.

Every experiment run on it, with its question, result and status: [`EXPERIMENTS.md`](EXPERIMENTS.md).

Korvane (`tests/corpus/korvane/`) is the hand-written corpus (80 files, 1,130 facts, reviewed by people). Quillmoor is
its large, generated counterpart: every fact is known to the generator, so every number below is checked mechanically
against ground truth, at any size.

## The world

**Quillmoor Utilities plc** runs tidal stations, water treatment works, pumping stations, desalination plants,
substations and reservoirs across the invented Sarne Isles. Thirteen entity types, each with typed attributes:

| Entity | Attributes stated in the files |
|---|---|
| site | code, island, commissioned, capacity, permit, manager, headcount, fire assembly point (pictures only), telemetry poll interval and historian retention (config files only) |
| person | employee ID, role, site, start date, extension (some changed later: the directory is outdated, the chat has the new number), certification, expiry |
| asset | model, serial, install date, rating, criticality, site, service interval, last overhaul, mass (nameplate photos only) |
| supplier | vendor no., category, country, payment terms, rating, contact |
| contract | counterparty, value, start, end, delay penalty, notice period, signatory (some amended: the contract states the old value, a variation the new one) |
| customer | account no., tariff, monthly volume, billing day, account manager |
| incident | date, site, asset, duration, root cause, cost, reporter, severity, alarm code (logs only), insurance claim ref (scanned letters only) |
| part | description, unit price, stock, reorder point, lead time, bin, supplier |
| policy | number, effective date, owner, 3 rules each (45 % revised: v1 old value, v2 + a change notice new value) |
| service | version, port, timeout (config), error codes and their meaning (source code), refresh interval (UI code), owner and SLA (runbook) |
| project | budget, sponsor, project manager, start, go-live, status |
| chemical | dose, storage temperature, hazard class, max stock, supplier |
| vessel | call sign, length, crew, home site, built (half in registers, half only in scanned harbour letters) |

Asset tags and part numbers come in near-duplicate clusters on purpose (`PMP-4471-B` and `PMP-4471-D` are different
pumps at different sites), so retrieval has to tell look-alikes apart.

## Files: 42 generated extensions

Quillmoor exercises 42 extensions; the parser registry has 43. `.msg` is deliberately excluded: Outlook messages use
an OLE Compound File structure ([Microsoft MS-OXMSG](https://learn.microsoft.com/en-us/openspecs/exchange_server_protocols/ms-oxmsg)), while the current app handler treats `.msg` as RFC822 text. It is not counted as
supported until a real OLE-format fixture is parsed correctly. The generated set includes image-only scanned PDFs:

- prose: `.docx .doc .odt .pdf .rtf .epub .html .htm .md .markdown .txt`
- tables: `.xlsx .xls .ods .csv .tsv` (multi-sheet workbooks split long registers), tables inside `.docx .pdf .pptx .html .md`
- records: `.json .jsonl .xml .yaml .yml`
- config: `.ini .cfg .conf .yaml .yml .json .xml`
- code: `.py .js .ts .tsx .jsx .css`
- logs and chat: `.log .txt .md .json`
- mail and slides: `.eml .pptx .odp`
- pictures (OCR): `.png .jpg .jpeg .tif .tiff .bmp .webp .gif`, and `.pdf` pages that are only images

Each document kind decides its structure (a register is a table, an incident report is a form + prose, a nameplate is a
photo), the generator picks one allowed type per document so all types are used about equally, and records the exact
string it wrote for every value. About 7 % of the files are distractors (staff news) with numbers but no facts.

The three deterministic tiers cover the same 29 archetypes; these are the generated file counts by structure:

| Archetype | S | M | L |
|---|---:|---:|---:|
| `asset_register` | 12 | 40 | 160 |
| `cert_matrix` | 12 | 40 | 160 |
| `chat_log` | 12 | 40 | 160 |
| `chemical_sds` | 12 | 36 | 120 |
| `contract_amendment` | 8 | 59 | 363 |
| `contract_doc` | 40 | 220 | 1,400 |
| `customer_accounts` | 3 | 18 | 130 |
| `distractor` | 27 | 128 | 693 |
| `incident_log` | 12 | 40 | 160 |
| `incident_report` | 50 | 360 | 2,600 |
| `insurance_scan` | 8 | 59 | 322 |
| `maintenance_plan` | 24 | 120 | 640 |
| `nameplate` | 33 | 168 | 974 |
| `parts_catalog` | 6 | 45 | 325 |
| `policy_doc` | 17 | 53 | 156 |
| `policy_notice` | 5 | 17 | 46 |
| `project_update` | 14 | 70 | 380 |
| `runbook` | 14 | 48 | 180 |
| `service_board` | 14 | 48 | 180 |
| `service_config` | 14 | 48 | 180 |
| `service_css` | 14 | 48 | 180 |
| `service_errors` | 14 | 48 | 180 |
| `site_config` | 12 | 40 | 160 |
| `site_notice` | 12 | 40 | 160 |
| `site_profile` | 12 | 40 | 160 |
| `staff_directory` | 5 | 24 | 120 |
| `supplier_list` | 4 | 12 | 53 |
| `vessel_register` | 1 | 3 | 10 |
| `vessel_scan` | 4 | 15 | 55 |

## Questions

`eval/questions.jsonl`, every one with its answer, accepted spellings, outdated (wrong) values and the evidence files:

| Type | What it tests |
|---|---|
| paraphrase | re-worded question, never used for training; sampled per file type, so accuracy is attributable to a format |
| verbatim | the training phrasing (memorisation; only meaningful for a model trained on `train/train.jsonl`) |
| superseded | the value changed: the newest is right, the older file's value is wrong |
| multihop | two facts from two files (asset → its site → the site's manager, part → supplier → payment terms, ...) |
| reverse | find the entity from a value (which asset has serial SN...?) |
| unanswerable | the files do not say (an attribute nobody wrote down, or a look-alike entity that does not exist) |

`train/train.jsonl` holds one chat row per fact in the direct phrasing with the current value: what a user would train
an adapter on. The `train_variants.py` recipe writes three prompts per fact to `train_x3_grouped.jsonl`, with
`pair_id=fact` so the app keeps all phrasings on the same side of its validation split. The earlier `train_x3.jsonl`
is row-wise and leaks 1,165 of 1,167 validation facts into training; it is retained only as evidence that the training
pipeline completed, never as a generalization score.

## Model families

The S-tier answer matrix uses these public Hugging Face model IDs converted to q8_0 GGUF by the app's converter;
each result is labelled by repo family in `RESULTS.md`. Training was attempted on a subset; exact attempts and valid
artifacts are listed there.

| Family | Hugging Face model |
|---|---|
| Qwen 3.5 0.8B | `Qwen/Qwen3.5-0.8B` |
| Qwen 3 1.7B | `Qwen/Qwen3-1.7B` |
| OLMo 2 1B | `allenai/OLMo-2-0425-1B-Instruct` |
| Granite 3.3 2B | `ibm-granite/granite-3.3-2b-instruct` |
| Llama 3.2 3B | `unsloth/Llama-3.2-3B-Instruct` |
| SmolLM3 3B | `HuggingFaceTB/SmolLM3-3B` |
| Phi-4 mini | `microsoft/Phi-4-mini-instruct` |
| Gemma 3 4B | `unsloth/gemma-3-4b-it` |
| Gemma 4 E2B | `google/gemma-4-E2B-it` |
| Qwen 3.5 4B | `Qwen/Qwen3.5-4B` |

## Tiers

| Tier | Files | Facts | Use |
|---|---|---|---|
| s | 415 | 4,306 | everything, including fine-tuning |
| m | 1,927 | 25,112 | ingest throughput, Smart Brain at scale |
| l | 10,407 | 157,614 | large-library ingest and index build |

Generation is deterministic for a tier and seed (text is byte-identical; office containers embed timestamps).

## Commands

```bash
# generate (files land in .tmp/, never committed)
.venv/bin/python tests/corpus/quillmoor/generate.py --tier s --out .tmp/quillmoor/s
.venv/bin/python tests/corpus/quillmoor/qmgen/render_selftest.py          # generated writers x representative app-parser fixtures

# benchmarks against the live app (an API token in .tmp/bench/token: fts auth token create bench)
B=tests/corpus/quillmoor/bench
.venv/bin/python $B/ingest.py --corpus .tmp/quillmoor/s --project agent-test-quillmoor-s
.venv/bin/python $B/brain_matrix.py --corpus .tmp/quillmoor/s --project agent-test-quillmoor-s \
    --embedders Qwen/Qwen3-Embedding-0.6B --rerankers none BAAI/bge-reranker-v2-m3
.venv/bin/python $B/base_gguf.py Qwen__Qwen3.5-4B:q8_0
.venv/bin/python $B/finetune.py --corpus .tmp/quillmoor/s --project agent-test-quillmoor-s --models Qwen__Qwen3.5-4B:q8_0
.venv/bin/python $B/train_variants.py --corpus .tmp/quillmoor/s
.venv/bin/python $B/finetune.py --corpus .tmp/quillmoor/s --project agent-test-quillmoor-s \
    --models Qwen__Qwen3-1.7B:q8_0 unsloth__Llama-3.2-3B-Instruct:q8_0 \
    --train-file train_x3_grouped.jsonl --tag x3-grouped --epochs 8 --rank 64
# run base and tuned GGUFs with distinct labels; both use the same 862 fact IDs
.venv/bin/python $B/evaluate.py --corpus .tmp/quillmoor/s --label qwen3-1.7b-base \
    --gguf <base-gguf> --modes closed --sft-facts train.jsonl --workers 1
.venv/bin/python $B/evaluate.py --corpus .tmp/quillmoor/s --label qwen3-1.7b-x3-grouped \
    --gguf <tuned-gguf> --modes closed --sft-facts train_x3_grouped.jsonl --workers 1
.venv/bin/python $B/evaluate.py --corpus .tmp/quillmoor/s --label qwen35-4b-base --gguf <gguf> --modes closed rag --package <dir>
.venv/bin/python $B/rag_answers.py --corpus .tmp/quillmoor/s --config default --package <dir> \
    --model qwen35-4b-base=<gguf> --model llama32-3b-base=<gguf>
.venv/bin/python $B/export_combos.py --corpus .tmp/quillmoor/s --project agent-test-quillmoor-s --answer-gguf <gguf>
.venv/bin/python $B/report.py --tiers .tmp/quillmoor/s .tmp/quillmoor/m .tmp/quillmoor/l
.venv/bin/python $B/report.py --tiers .tmp/quillmoor/s .tmp/quillmoor/m .tmp/quillmoor/l > tests/corpus/quillmoor/RESULTS-TABLES.md
```

Projects are named `agent-test-*` so `.tmp/agent/cleanup.py` removes them. Results land in `<corpus>/results/`;
after training, rerun the same `finetune.py` command with `--refresh-metrics` to copy actual optimizer steps, dataset
rows, and GGUF sizes from live run records without starting another training job. The measured numbers of record are in
`RESULTS.md` and the complete generated tables in `RESULTS-TABLES.md` next to this file, with the interpretation and
limitations in `docs/benchmarks/`. `evaluate.py --sft-facts <train-file>` compares a deterministic sample of facts
seen by training (seed `20261029`) with every fact in the app's validation split, refuses a leaking split, and writes answer/judge
artifacts separately under `results/sft-facts/`. The judge checks recall on seen facts and expects abstention on
fictional held-out facts without showing their answer key; the generated report checks exact source IDs and separate
answer/judge coverage for both groups.
