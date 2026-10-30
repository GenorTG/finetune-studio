# How Finetune Studio works

*A technical overview for people deciding whether, and how, to use it. Every claim about results links to a measured
run; see [the Quillmoor benchmark report](quillmoor-2026-10.md) and the Korvane
[RESULTS.md](../../tests/corpus/korvane/RESULTS.md).*

## 1. One project, one parse, two ways to use it

Everything happens inside a **project**. A project owns its files, the text parsed from them, the question/answer
pairs mined from that text, training datasets, training runs, tests, exports and a Smart Brain index.

```
files ──► Files & Knowledge (parse once, check, edit)
              │
              ├──► Smart Brain: passages ─► index (vectors + BM25) ─► search ─► rerank ─► answer with sources
              │                                        └─► export: standalone server (HTTP + MCP) / .ftsrag bundle
              │
              └──► Pairs: mine Q&A per passage ─► review ─► dataset ─► Training (LoRA: SFT / DPO / KTO)
                                                                          └─► Testing (AI judge) ─► Export (GGUF, merged, ...)
```

The two routes answer different needs:

| | Smart Brain (retrieval) | Adapter (fine-tuning) |
|---|---|---|
| What it changes | what the model *reads* before answering | the model's *weights* |
| Good at | facts, numbers, names, anything that must be exact and cited; documents that change | tone, format, a task (classify, extract), a house style, a fixed system prompt |
| Bad at | changing how the model behaves | carrying hundreds of exact facts across re-worded questions; saying "I don't know" |
| Cost to update | rebuild the index (seconds to minutes) | train again (minutes to hours, a GPU) |

They combine: a fine-tuned model can be the answer model of a Smart Brain.

## 2. Files & Knowledge

Every upload is stored as raw bytes (immutable, de-duplicated by SHA-256) and parsed **once** into text that every
later step reads. The parser registry lists 43 filename extensions, including plain text and code, CSV/TSV,
JSON/JSONL/XML/YAML, HTML, Markdown, PDF (text layer and scanned-page OCR), Word (`.docx`/`.doc`), Excel (`.xlsx`/`.xls`),
OpenDocument (`.odt/.ods/.odp`), PowerPoint, RTF, EPUB, e-mail (`.eml`), and pictures through Tesseract OCR (plus an
optional vision model that reads charts and photos). Outlook `.msg` is listed, but its current handler treats it as
RFC822 text rather than parsing the binary MSG container; do not count `.msg` as supported pending a parser fix. Tables
keep their header with every row, so a row cut into a passage still says what its numbers are. The Quillmoor generator
exercises 42 extensions and has no `.msg` writer. Text-layer PDFs and scanned PDFs are reported as separate ingest buckets.

The parsed text is shown next to the original and can be corrected by hand; the correction feeds mining and the next
index build while the raw file stays untouched.

**Measured** (Quillmoor M tier, 2026-10-29): 1,927 synthetic files across 42 generated extensions and 29
document archetypes, 145.98 MB;
upload took 137.2 s and parsing finished 61.8 s later, with 0 parse failures. The parser found 25,921/25,957 known
values (99.86%); all non-OCR values matched. A re-read using identifier boundaries (hyphens count as identifier
characters) found the same 25,921/25,957; this still does not test field boundaries. OCR found 810/846 marked values
(95.74%): 35 missed values are equipment ratings written with the superscript unit `m³/h`, and one is a call sign. These generated fixtures are representative
engineering checks, not evidence for every real-world document producer or layout.

## 3. Smart Brain

### Index
Each parsed file is cut into **passages** (paragraph- and table-row-aware). A build embeds every passage with a
sentence-transformers model (any Hugging Face embedder; the default is `intfloat/multilingual-e5-large`) and builds a
BM25 keyword index over the same passages. Builds run as their own job process with live progress, can be stopped,
and swap the new index in only when it is complete.

### Search
A question goes through up to four layers, each one switchable:

1. **Meaning search** — cosine similarity between the question's vector and the passages' vectors.
2. **Word search** — BM25 over the passages' words (exact identifiers, part numbers, names).
3. **Rank fusion** — when both run, their rankings are merged by Reciprocal Rank Fusion (RRF).
4. **Reranker** — a cross-encoder reads the question together with each of the best candidates (50 by default) and
   re-orders them. A new index gets bge-reranker-v2-m3 (multilingual, run in half precision) on a machine with a GPU
   and ms-marco-MiniLM-L-6-v2 (small, English) without one.

Then the **answer layer** gives the best passages to a chat model with a grounded prompt ("answer from these
passages, cite them, say so when they do not contain the answer"). Any OpenAI-compatible endpoint works: a local
llama.cpp server, an API, or the studio's own model. When the passages lack a fact the question needs — typically the
second step of "who manages the site where asset X is installed?" — the model may reply `SEARCH: <query>` instead;
that search runs, its passages join the first ones, and the model answers (one follow-up by default, `RAG_FOLLOW_UP`
in an exported package).

### Export
- **Standalone package**: a self-installing directory (`install.sh`, `run-http.sh`, `run-mcp.sh`) with a small
  server (HTTP `/search`, `/ask`, `/health`, and MCP tools `rag_search`, `rag_answer`). It ships only the files of
  the layers you keep (no vectors without meaning search, no BM25 index without word search), optionally the embedder
  and reranker models (fully offline), and is **encrypted by default** (AES-256-GCM, scrypt key from a passphrase
  that is shown once and never stored). The answer model's API key is never put in the package.
- **`.ftsrag` bundle**: an encrypted copy of the index to load into another studio's project.

## 4. Adapters

### Pairs
The AI assistant (any OpenAI-compatible model, or a local GGUF) reads each passage and writes question/answer pairs;
checks reject answers that are not in the source, duplicates and trivial pairs. The Pairs page is a review queue
(keyboard driven, the source passage next to every pair). Approved pairs become a dataset; a dataset can also be
uploaded (JSONL / JSON / CSV) or built from the reviewed set plus optional re-worded copies (study set) and
"I don't know" rows.

### Training
LoRA on any safetensors chat model the GPU fits, through TRL (Unsloth when installed): SFT, DPO, KTO, tool-use SFT,
continued pre-training and reasoning distillation. The base is loaded in 4-bit (QLoRA) when bitsandbytes works on the
GPU. The seeded validation split holds out about 10 % of its split units: examples with the same `pair_id` stay
together, while rows without a group ID are split individually. If paraphrases or repeated examples should count as
one fact, the dataset must set `pair_id` to that fact's stable ID; otherwise the same fact can occur on both sides of
the split. The Quillmoor report measures that overlap explicitly. A run is its own job process: it survives a web
restart, reports live loss and ETA, and can merge the adapter and export GGUF quantisations in the same job.

### Testing
A test run records questions, the model's answers and the answer key. **Correctness is decided afterwards by an AI
judge** (any provider; prompt and rubric in `testing/judge.py`), never by string matching, and a person can override
any verdict. Unanswerable questions are graded on whether the model declines.

### Export
Adapter only, merged safetensors, GGUF in any llama.cpp quantisation (optionally with an importance matrix computed
from the training text), and two refusal-removal variants: **Abliterated** (one fixed direction) and **Heretic**
(an optional external tool that searches the abliteration parameters within a KL-divergence budget).

## 5. What runs where

- The web app (FastAPI) runs under a small supervisor; GPU work (training, index builds, exports, model loads) runs
  in separate job processes that take a GPU lease, so two GPU jobs never fight over memory and a web restart does
  not end them.
- One GPU is chosen in Settings → Compute device; nothing else on the machine is stopped to make room.
- Everything stays on the machine unless you configure an API connection for the AI assistant or the answer layer.

## 6. Where to read the numbers

- [Quillmoor benchmark (generated corpus across 42 extensions; measured model-family results)](quillmoor-2026-10.md)
- [Choosing a mode: what to expect from each mode and mix](choosing-a-mode.md)
- [Korvane RESULTS (hand-written corpus, reviewed pairs, Qwen3.5-9B)](../../tests/corpus/korvane/RESULTS.md)
