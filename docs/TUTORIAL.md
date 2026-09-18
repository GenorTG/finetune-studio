# Tutorial — your first model, start to finish

A complete hands-on walkthrough of Finetune Studio: from a folder of raw
documents to a fine-tuned, exported model you can chat with and evaluate.
Every step happens in the browser on your own machine.

Prerequisites: the app is running (see [INSTALL.md](INSTALL.md)) at
`http://localhost:7860`. A CUDA GPU with ≥12 GB VRAM is recommended for
Qwen3-4B-class training; RAG and chat work CPU-only too.

---

## The core loop

```
Upload → Parse → QA pairs → Approve → Export → Train → Benchmark → Export → Chat
```

Everything else (RAG, testing, production runs) hangs off this spine.

### 0. Bootstrap (once)

1. Open `http://localhost:7860`. The dashboard loads; the first visit
   starts an 8-step onboarding tour (replayable anytime: Tools → Settings
   → Replay Tutorial).
2. Click `+ New Project`, give it a name. Projects own everything: files,
   datasets, corpora, runs, exports, suites.

### 1. Get raw documents in (Data Prep)

1. Open your project → **Data Prep** tab.
2. Drag files into the **Uploaded files** library (or ⬆ Upload). Anything
   readable is parsed to text: PDF, DOCX, DOC, TXT, MD, HTML, code, CSV,
   JSON, EPUB, RTF, and more. Files are SHA-256 deduped and stored
   immutable; parsing never mutates the raw upload.
3. Click **Parse** (or 📄 actions per file). Parsed files become **Parsed
   sources** — the text layer every later stage reads from. A placeholder
   parse (for example, a .doc needing a missing dependency) is retried
   automatically; open 📝 to preview the converted markdown.

### 2. Turn sources into training pairs (QA mining)

Three ways to get prompt/response training data out of your parsed
sources:

- **Prep job (no LLM)** — extractive Q&A: questions built from real
  sentences in the sources; zero hallucination risk, fastest start.
- **Local Agent (Chat tab, Agent mode)** — the helper LLM reads chunks and
  drafts high-quality pairs; you approve in bulk.
- **Manual upload** — bring your own JSONL
  (`{"messages":[{role,content},…]}` per line).

Then **triage**: review pairs, ✅ approve / ✗ reject (bulk actions
supported). Rejected identifiers (IDs, paths) stay out of training so the
model doesn't learn to parrot internal keys.

### 3. Export pairs → dataset

Click **Export approved → Training**. This writes a ShareGPT JSONL and
registers it under the project. A green **Start training with this
dataset →** button appears — that's your handoff; it opens Training with
the dataset preselected. Check the JSONL any time in Data Editor.

### 4. Train (Training tab)

1. **Base model** — pick a local transformers-compatible base (GGUF/GPTQ
   exports are inference-only; they can't be trained on).
2. **Training data** — *From this project* is your exported dataset;
   *Upload my own* registers a raw JSONL.
3. Set epochs, LoRA rank, learning rate, batch size. For a small factual
   domain dataset, plan for enough real optimizer steps — a handful of
   epochs at 4× gradient accumulation is not memorization (see the badges
   on the run: they report actual steps, never fake progress).
4. **Start training.** Live loss/step/LR stream over SSE; the activity feed
   logs every REST call. VRAM is profiled before the run and an
   actionable error tells you what's holding memory if it can't fit.
5. Runs can be stopped; interrupted runs are reconciled at next startup.
   Past-run rows carry ⭐ Set production / ▶ Inference / ⬇ Download.

### 5. Export (Export tab)

Pick a run, pick formats:

- **merged** — adapter merged onto a 16-bit base (required before GGUF if
  you trained on a 4-bit/nf4 base; the export path auto-merges and any
  failure is written to the run row, not silently swallowed).
- **GGUF** — q4_K_M … q8_0, needs llama.cpp conversion tools on the host;
  a clear install hint otherwise. Never a fake success.
- **abliterated** — refusal-direction edit of a merged checkpoint.
- **GPTQ** — needs `auto-gptq`; honest failure if missing.

Exported artifacts show up under **Models** with expand rows (parent run,
settings, directory listing) and ▶ Open in inference.

### 6. Evaluate (Benchmarks tab)

- **Synthetic offline suites** (MMLU/GSM8K/HellaSwag-*shaped*) — fast,
  local, no downloads; good for smoke checks.
- **Project QA suite** — generated from your own held-out pairs; this is
  the one that tells you whether your model *memorized your domain*
  (with strict substring judging you can eyeball case-by-case under
  **Testing**).
- **Real industry suites** — official GSM8K/MMLU/HellaSwag splits,
  downloaded on first use; strict MCQ scoring. Bounded `num_samples` for
  fast runs, `full_run=true` via the API for full splits.
- **Compare two runs** tab — per-suite Δ table, colored.

Rule of thumb: if the auto-score says pass, spot-check 10–20 answers on
the Testing page. The judge is heuristic; your eyes are the gold standard
(see docs/judging/PROTOCOL.md).

### 7. Chat (Chat tab)

Load a safetensors **or GGUF** model, chat with streaming, image input for
multimodal models, per-session temperature/top-p/system prompt. History
persists across reloads; idle models auto-unload to free VRAM.

No model loaded yet? Use the red ⚠ panel's one-click loader right on the
page — no need to round-trip to Inference.

### 8. RAG (RAG tab) — optional

Corpus is separate from training: parse sources on Data Prep, then
**Embed** here to build chunks + BM25/dense index (parquet-backed, on
disk). Retrieval test shows what a query would fetch; chat with "RAG
mode" grounds answers in retrieved chunks instead of fine-tuned weights.
**Use RAG for facts that change; use fine-tuning for style/response
shape.** Both work against the same parsed sources.

---

## Common first-run problems

| Symptom | It means | Fix |
|---|---|---|
| "No datasets yet" in Training | Export step didn't run/complete | Data Prep → Export approved → Training; look for green success + the CTA |
| Chat says "no model loaded" | Nothing loaded in inference | Use the ⚠ panel's Load dropdown on the chat page itself |
| Training starts but feels instant-and-done | Loop returned without real steps | Check the run badge's optimizer-step count; raise epochs / dataset size |
| Export says merge needed | Adapter trained on 4-bit base | Set base model to the 16-bit sibling (e.g. `Qwen/Qwen3-0.6B`), not nf4 |
| GGUF button disabled | llama.cpp tools missing | Install llama.cpp; the API error says exactly what |
| Chat unrelated answers | No/weak system prompt + wrong production run | Training → ⭐ Set production on the good run; Chat → pick the project's system prompt |

## Where everything lives on disk

```
~/.finetune-studio/
├── projects/{pid}/          project data, suites, DB-backed metadata
├── rag_corpora/{pid}/       parquet + per-source text
├── hf_models/               downloaded HF snapshots
└── shared_models/           embedder/reranker weights shared across corpora
output/projects/{pid}/runs/  checkpoints, merged/, gguf/, logs
```

Everything is files on your disk — no invisible cloud copies, no lock-in.

## Where to go next

- [README.md](../README.md) — the tour of all pages
- [ARCHITECTURE.md](ARCHITECTURE.md) — how the pieces fit
- [judging/PROTOCOL.md](judging/PROTOCOL.md) — human-grade score verification
- [DEPENDENCIES.md](DEPENDENCIES.md) — every dependency and why it's there
