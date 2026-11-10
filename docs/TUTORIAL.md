# Tutorial — your first model, start to finish

A complete hands-on walkthrough of Finetune Studio: from a folder of raw
documents to a Smart Brain that answers from them and a LoRA adapter that answers in your voice, exported and evaluated.
Every step happens in the browser on your own machine.

Prerequisites: the app is running (see [INSTALL.md](INSTALL.md)) at
`http://localhost:7860`. A CUDA GPU with ≥12 GB VRAM is recommended for
Qwen3-4B-class training; the Smart Brain and chat work CPU-only too.

---

## One main path, one optional one

Every file is parsed once in **Files**. From there the main path is the Smart Brain; training a model on the same
files is optional and lives in its own group at the end of the project bar:

```
1 files → 2 Smart Brain → 3 test → 4 package        (minutes, no training)
optional: quick work · pairs · train · compare · benchmarks
```

The project home shows a live checklist and one suggested next action, with training linked as the optional path, and the sub-nav lists the numbered steps of whichever flow you're in. You can do both, in any order — they never block each other.

**The short path: ⚡ Quick work.** One page (`work` in the sub-nav) holds the
whole model pipeline as six numbered cards — upload, make pairs, build the
dataset, train, test, pin — each with a status pill and one button. There's
even a **Run all training steps** chain that drives the entire pipeline unattended
while you watch. Use the individual pages when you want fine control; use
Quick work when you just want the model.

### 0. Bootstrap (once)

1. Open `http://localhost:7860`. The first visit starts a short, task-oriented onboarding tour; replay it any time from **Settings → Replay**.
2. Click `＋ New project`, give it a name. Projects own everything: files,
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
   (pairs → datasets → runs → Smart Brain).
4. Want to fix a bad parse? Open a file's 📝 editor and edit the **parsed
   text** — the correction feeds Q&A mining and the next Smart Brain build, while
   the original upload stays byte-identical.
5. Every readable upload is parsed **once, on arrival** — that parse is the text layer every later stage
   reads (training pairs and the Smart Brain alike). Tick **search inside files** next to the search box to search
   the mined facts and the parsed text, and fix a fact right there.

### 2. Smart Brain — answer from your files (step **2 · smart brain**)

Independent of training; same parse. The Smart Brain finds the passages that answer a question (semantic +
keyword search, rank fusion, reranking — often called RAG) and the model quotes them. On its page:

1. Add files on the page itself (**＋ Add files**) or in **Files**.
2. **Build Smart Brain** (Quick work's
   Step 2 card does the whole flow in one click and shows a live coverage
   pill, e.g. "129/129 · 100 %"). Sources are chunked and indexed for
   keyword + meaning search; the first build may download the embedder. The panel at the top says *Ready* or
   *Needs an update* when files were edited, added or removed since. Optional **fact cards** (under **Build options**)
   also index your approved Q&A pairs.
3. **Documents in the index** — what actually got indexed; a missing file
   wasn't parsed.
4. **Test a search** — a real query with per-passage scores, no AI answer, so
   you can judge retrieval alone.
5. **Ask your documents** — chat over those passages; follow-up questions keep the conversation, and each answer lists the files it used. **Answer with**
   picks who writes the answer: the chat model loaded on this machine (the
   default; the top-bar pill says which; nothing leaves the machine), or, as a
   deliberate choice, the AI assistant's API connection from *Settings → AI
   assistant* (needs no local model; the question and the retrieved passages
   are sent to that provider). It is never picked for you: with no local model
   loaded the page offers to load one right there, or you can choose the assistant.
6. **Share or run it elsewhere** — the encrypted studio bundle (.ftsrag), or the **standalone
   Smart Brain**: one tarball with the index + a small Python server +
   `setup.sh`. On a machine with Python 3 (`python3` with the `venv` module; the tests run its server on 3.12):
   `bash setup.sh` installs a venv, picks a port, starts the server, can install it as a persistent
   service, and prints a copy-paste MCP entry. It speaks **MCP** (Claude
   Desktop / OpenClaw / Cursor) and plain **HTTP** (`GET /search?q=…`).
   Tick "Include models" (about 0.2 GB with the small MiniLM models, about 4.5 GB with
   e5-large and bge-reranker-v2-m3) and the embedder + reranker ship inside
   the package: full offline semantic search with no external service.

**Use the Smart Brain for facts that change; use fine-tuning for style/response shape.**
Both read the same parse.

---

### 3. Test it (step **3 · test**)

The **Testing** page answers one question: *did it learn MY material?*
Auto-generated project suites test **every row of the training dataset**
(N rows → N questions, full coverage by default; sampling is an explicit,
labeled opt-in). A test is two steps: **Run** saves the model's raw answers next
to the correct answers; **Judge** lets an AI judge (any connected model; by default the AI assistant's
*Test judge* job) or you decide pass / partial / fail for each saved answer
(*Settings → Test judge* or the Testing page's *judge automatically after each run* switch can start judging
automatically after every run; it is off by default). Rule of thumb: spot-check 10–20 answers with your own eyes — the
judge is a model, not truth — and press 1/2/3 in the review pane to override a
verdict case by case.

**Benchmarks** (per project, optional) is separate: public exams — the official MMLU / GSM8K /
HellaSwag test sets from Hugging Face, or offline synthetic look-alikes — and a **Compare two runs**
Δ table with a per-question McNemar test. Useful, and not a substitute for the project test.

### 4. Use it (step **4 · package**)

- **Pin a version** — Quick work's Step 6 pins the exact dataset + corpus +
  run + base model into an immutable project version, so any result is
  reproducible and branchable.
- **Export** (project **export** page): **merged** (adapter onto a 16-bit
  base — auto-merged before GGUF; failures land on the run row, never
  silent), **GGUF** (q4_K_M … q8_0, needs llama.cpp tools on the host; tick **importance matrix**
  for better low-bit quality, measured on the run's own training data), **abliterated**
  (refusal-direction edit of a merged checkpoint; offered, not yet run end to end on a real model), and
  **Heretic** (an optional separate tool, installed from *Settings → Heretic*, AGPL-3.0, that searches the
  refusal-removal parameters to keep the model close to its original behaviour).
- **Models** page: every export in one table — click a row to expand the
  parent run, settings, and directory contents; ▶ Open in inference.
- **Chat**: load a safetensors **or GGUF** model, stream, image input for
  multimodal, per-session temperature/top-p/system prompt. History
  persists; idle models auto-unload. No model loaded? The red ⚠ panel has
  a one-click loader right on the page.

## Optional: train a model on the same files

For how a model answers (tone, format, a narrow task). For facts in your documents, the Smart Brain above is more accurate. These are the **Train a model** pages at the end of the project bar.

### 5. Turn sources into training pairs (the optional **pairs** page)

Ways to get prompt/response training data out of your parsed sources:

- **Mine pairs** (the **pairs** page) — the AI assistant (the local model or an API connection you
  added in *Settings → AI assistant*) works through each file: a checklist of its data points, pairs
  written for them, checked against the text and fixed until every point is stated or excluded.
- **Ask the Guide** (the panel in the corner) — it can start the same mining job for you and tell you
  what is left; it asks before it runs anything unless you switch it to hands-free.
- **Manual upload** — bring your own JSONL
  (`{"messages":[{role,content},…]}` per line).
- **A use case** (Training page → *Start from a use case*) — extraction, classification, house style
  or distillation examples, written by you or drafted by the assistant from raw texts for you to approve.

Then **triage**: review pairs, ✅ approve / ✗ reject (bulk actions
supported). Rejected identifiers (IDs, paths) stay out of training so the
model doesn't learn to parrot internal keys.

### 6. Export pairs → dataset

Click **Export approved → Training**. This writes a ShareGPT JSONL and
registers it under the project. Before it writes anything, the completeness
gate checks every file's data-point checklist: each point must be stated by an
approved pair or excluded with a reason. An incomplete file stops the export
and lists its open points; you can fill them, exclude them, or choose
**export anyway**. Nothing is generated for you at export. This checks the
dataset, not the model: it does not guarantee that the trained model learned or
can reproduce every fact.

Two optional builders on the same page write extra datasets from your approved
pairs with the AI assistant: **Study set** (re-worded questions, re-said answers
and a short passage per pair, for an adapter that must recall facts) and
**Preference pairs (DPO)** (a chosen source-faithful or honest “not covered”
answer next to a rejected guess, for the Preference tuning route).
A green **Start training with this dataset →** button appears — that's your
handoff; it opens Training with the dataset preselected.

### 7. Train (the optional **train** page)

1. **Base model** — pick a local transformers-compatible base (GGUF
   exports are inference-only; they can't be trained on). The preset picker
   recommends rank/LR/epochs from base + dataset size.
2. **Training data** — *From this project* is your exported dataset;
   *Upload my own* registers a raw JSONL.
3. Set epochs, LoRA rank, learning rate, batch size. For a small factual
   domain dataset, plan for enough real optimizer steps — a handful of
   epochs at 4× gradient accumulation is not memorization (see the badges
   on the run: they report actual steps, never fake progress).
4. **Start training.** Live loss/step/LR stream over SSE; the activity feed
   logs every change you make. VRAM is profiled before the run and an actionable
   error tells you what's holding memory if it can't fit.
5. Runs can be stopped; interrupted runs are reconciled at next startup.
   Past-run rows carry Details / ⭐ Set production / ▶ Open in inference / ⬇ Export.

## Common first-run problems

| Symptom | It means | Fix |
|---|---|---|
| "No datasets yet" in Training | Export step didn't run/complete | **pairs** → Export approved → Training; look for the green success + CTA |
| Chat says "no model loaded" | Nothing loaded in inference | Use the ⚠ panel's Load dropdown on the chat page itself |
| Training starts but feels instant-and-done | Loop returned without real steps | Check the run badge's optimizer-step count; raise epochs / dataset size |
| Export says merge needed | Adapter trained on 4-bit base | Set base model to the 16-bit sibling (e.g. `Qwen/Qwen3-4B`), not nf4 |
| GGUF button disabled | llama.cpp tools missing | Install llama.cpp; the API error says exactly what |
| Chat unrelated answers | No/weak system prompt + wrong production run | Training → ⭐ Set production on the good run; Chat → pick the project's system prompt |
| Smart Brain search finds nothing for a file | It was never indexed, or it changed since the build | **files** → check the pipeline badges; then Smart Brain → ⚡ Quick index |

## Where everything lives on disk

```
~/.finetune-studio/
├── projects/{pid}/          project files, their parse, suites
├── brain/{pid}/             Smart Brain index: parquet + vectors + BM25 + per-source text
├── hf_models/               downloaded HF snapshots
└── shared_models/           embedder/reranker weights shared across indexes
<checkout>/data/finetune_studio.db   the project database (projects, datasets, runs, tests)
<checkout>/output/projects/{pid}/runs/  checkpoints, merged/, gguf/, logs
```

Project content is stored in the configured local data roots. Optional
external model, embedding, or integration services may receive requests you
send to them; choose local endpoints when data must stay on your host.

## Where to go next

- [README.md](../README.md) — the tour of all pages
- [INSTALL.md](INSTALL.md) — install and troubleshooting
- [DEPENDENCIES.md](DEPENDENCIES.md) — every dependency and why it's there
