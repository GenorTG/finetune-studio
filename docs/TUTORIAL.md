# Tutorial — your first model, start to finish

A complete hands-on walkthrough of Finetune Studio: from a folder of raw
documents to a fine-tuned, exported model you can chat with and evaluate.
Every step happens in the browser on your own machine.

Prerequisites: the app is running (see [INSTALL.md](INSTALL.md)) at
`http://localhost:7860`. A CUDA GPU with ≥12 GB VRAM is recommended for
Qwen3-4B-class training; RAG and chat work CPU-only too.

---

## Two flows, one project

Every project offers two independent paths over the **same uploaded files**:

```
🧠 Train a model    1 files → 2 pairs → 3 train → 4 test → 5 use it
📚 Search my files  1 files → 2 build → 3 search → 4 chat        (RAG — minutes, no training)
```

The project home shows these as two big cards so you always know where to
start, and the sub-nav lists the numbered steps of whichever flow you're in.
You can do both, in any order — they never block each other.

**The short path: ⚡ Quick work.** One page (`work` in the sub-nav) holds the
whole model pipeline as six numbered cards — upload, make pairs, build the
dataset, train, test, pin — each with a status pill and one button. There's
even a **Run steps 2–5 now** chain that drives the entire pipeline unattended
while you watch. Use the individual pages when you want fine control; use
Quick work when you just want the model.

### 0. Bootstrap (once)

1. Open `http://localhost:7860`. The dashboard loads; the first visit
   starts an 8-step onboarding tour (replayable anytime: Tools → Settings
   → Replay Tutorial).
2. Click `+ New Project`, give it a name. Projects own everything: files,
   datasets, corpora, runs, exports, versions, suites.

### 1. Get raw documents in (step **1 · files**)

1. Open your project → **files** (or Quick work's Step 1 card).
2. Drag files in (or ⬆ Upload). Anything readable is parsed to text: PDF,
   DOCX, DOC, TXT, MD, HTML, code, CSV, JSON, EPUB, RTF, and more. Files
   are SHA-256 deduped and stored **immutable** — parsing and editing never
   touch the raw bytes.
3. The file browser is the hub: folders (drag files between them), tags,
   bulk actions, column picker, thumbnails, pagination, a 7-day trash with
   restore, ⬇ zip export, and an ℹ️ usage view showing where each file went
   (pairs → datasets → runs → RAG).
4. Want to fix a bad parse? Open a file's 📝 editor and edit the **parsed
   text** — the correction feeds Q&A mining and the next RAG build, while
   the original upload stays byte-identical.
5. Parsed files become **sources** — the text layer every later stage reads.

### 2. Turn sources into training pairs (step **2 · pairs**)

Three ways to get prompt/response training data out of your parsed sources:

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
registers it under the project. **The 100 % guarantee:** any chunk mining
missed is filled with verbatim extractive pairs at export time, so every
parsed chunk of every source is present in the dataset — no silent holes.
A green **Start training with this dataset →** button appears — that's your
handoff; it opens Training with the dataset preselected.

### 4. Train (step **3 · train**)

1. **Base model** — pick a local transformers-compatible base (GGUF/GPTQ
   exports are inference-only; they can't be trained on). The preset picker
   recommends rank/LR/epochs from base + dataset size.
2. **Training data** — *From this project* is your exported dataset;
   *Upload my own* registers a raw JSONL.
3. Set epochs, LoRA rank, learning rate, batch size. For a small factual
   domain dataset, plan for enough real optimizer steps — a handful of
   epochs at 4× gradient accumulation is not memorization (see the badges
   on the run: they report actual steps, never fake progress).
4. **Start training.** Live loss/step/LR stream over SSE; the activity feed
   logs every REST call. VRAM is profiled before the run and an actionable
   error tells you what's holding memory if it can't fit.
5. Runs can be stopped; interrupted runs are reconciled at next startup.
   Past-run rows carry ⭐ Set production / ▶ Inference / ⬇ Download.

### 5. Test it (step **4 · test**)

The **Testing** page answers one question: *did it learn MY material?*
Auto-generated project suites test **every row of the training dataset**
(N rows → N questions, full coverage by default; sampling is an explicit,
labeled opt-in). Results are strict-substring judged and reviewable
case-by-case. Rule of thumb: if the auto-score says pass, spot-check 10–20
answers with your own eyes — the judge is heuristic (see
[docs/judging/PROTOCOL.md](judging/PROTOCOL.md)).

**Benchmarks** (global page) is separate: public exams — offline smoke
suites styled after MMLU/GSM8K/HellaSwag, real HF splits when you allow
downloads, and a **Compare two runs** Δ table. Useful, optional, and not a
substitute for the project test.

### 6. Use it (step **5 · use it**)

- **Pin a version** — Quick work's Step 6 pins the exact dataset + corpus +
  run + base model into an immutable project version, so any result is
  reproducible and branchable.
- **Export** (project **export** page): **merged** (adapter onto a 16-bit
  base — auto-merged before GGUF; failures land on the run row, never
  silent), **GGUF** (q4_K_M … q8_0, needs llama.cpp tools on the host),
  **abliterated** (refusal-direction edit of a merged checkpoint), **GPTQ**
  (needs `auto-gptq`; honest failure if missing).
- **Models** page: every export in one table — click a row to expand the
  parent run, settings, and directory contents; ▶ Open in inference.
- **Chat**: load a safetensors **or GGUF** model, stream, image input for
  multimodal, per-session temperature/top-p/system prompt. History
  persists; idle models auto-unload. No model loaded? The red ⚠ panel has
  a one-click loader right on the page.

### 7. RAG — search my files (the other flow)

Independent of training; same parsed sources. In sub-nav order:

1. **1 · files** — upload + parse (there is no separate RAG upload).
2. **2 · build** — press **Build the index** (or ⚡ Quick index; Quick work's
   Step 1b card does the whole flow in one click and shows a live coverage
   pill, e.g. "129/129 · 100 %"). Sources are chunked and indexed for
   keyword + meaning search; the first build may download the embedder.
3. **Documents in the index** — what actually got indexed; a missing file
   wasn't parsed.
4. **Search test** — a real query with per-passage scores, no AI answer, so
   you can judge retrieval alone.
5. **Ask the model (grounded)** — chat over those passages (needs a chat
   model loaded — the top-bar pill says which).
6. **Download / export** — the plain corpus archive, or the **standalone
   package**: one tarball with the index + a small Python server +
   `setup.sh`. On any machine with Python 3.10+: `bash setup.sh` installs a
   venv, picks a port, starts the server, can install it as a persistent
   service, and prints a copy-paste MCP entry. It speaks **MCP** (Claude
   Desktop / OpenClaw / Cursor) and plain **HTTP** (`GET /search?q=…`).
   Tick "Include models" (~2.3 GB) and the embedder + reranker ship inside
   the package: full offline semantic search with no external service.

**Use RAG for facts that change; use fine-tuning for style/response shape.**
Both work against the same parsed sources.

---

## Common first-run problems

| Symptom | It means | Fix |
|---|---|---|
| "No datasets yet" in Training | Export step didn't run/complete | **pairs** → Export approved → Training; look for the green success + CTA |
| Chat says "no model loaded" | Nothing loaded in inference | Use the ⚠ panel's Load dropdown on the chat page itself |
| Training starts but feels instant-and-done | Loop returned without real steps | Check the run badge's optimizer-step count; raise epochs / dataset size |
| Export says merge needed | Adapter trained on 4-bit base | Set base model to the 16-bit sibling (e.g. `Qwen/Qwen3-4B`), not nf4 |
| GGUF button disabled | llama.cpp tools missing | Install llama.cpp; the API error says exactly what |
| Chat unrelated answers | No/weak system prompt + wrong production run | Training → ⭐ Set production on the good run; Chat → pick the project's system prompt |
| RAG search finds nothing for a file | It was never indexed | **files** → check the pipeline badges; then **build** → ⚡ Quick index |

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
