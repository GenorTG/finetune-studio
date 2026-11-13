# Tutorial — your first knowledge package, start to finish

A complete hands-on walkthrough of Finetune Studio: from a folder of raw documents to a checked, tested Smart Brain
that answers from them with sources, exported as one encrypted package any model can use over MCP or HTTP. Training a
LoRA adapter that answers in your voice is the optional second half. Every step happens in the browser on your own
machine.

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
2. Click `＋ New project`, pick the **Smart Brain** template, give it a name and press Create: the new project opens.
   Projects own everything: files, the Smart Brain, tests, packages, and (if you train) datasets, runs and exports.

### 1. Get raw documents in (step **1 · files**)

1. Open your project → **files** (or Quick work's Step 1 card).
2. Drag files or a whole folder in (or ⬆ Upload → *or a whole folder*; sub-folders are kept, so a `public/` and an
   `internal/` folder can later become two packages). Anything readable is parsed to text: PDF,
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
2. **Knowledge check** — finds unreadable files, duplicates, out-of-date versions (v3 next to v4, an older sheet kept in
   a workbook), documents that disagree and leaked passwords or keys. An amendment is linked to the agreement it amends.
   Remove an old version (or accept it), then press **Check again**: the package will not build with out-of-date files
   nobody accepted.
3. **Identity** — your company, its personality and how it should answer ("Lead with the answer. If the documents do
   not say, say so."). It goes with every answer and test and is packed into the export; no rebuild needed.
4. **Build Smart Brain** (Quick work's
   Step 2 card does the whole flow in one click and shows a live coverage
   pill, e.g. "129/129 · 100 %"). Sources are chunked and indexed for
   keyword + meaning search; the first build may download the embedder. The panel at the top says *Ready* or
   *Needs an update* when files were edited, added or removed since. Optional **fact cards** (under **Build options**)
   also index your approved Q&A pairs.
5. **Documents in the index** — what actually got indexed; a missing file
   wasn't parsed.
6. **Test a search** — a real query with per-passage scores, no AI answer, so
   you can judge retrieval alone.
7. **Ask your documents** — chat over those passages; follow-up questions keep the conversation, and each answer lists the files it used. **Answer with**
   picks who writes the answer: the chat model loaded on this machine (the
   default; the top-bar pill says which; nothing leaves the machine), or, as a
   deliberate choice, the AI assistant's API connection from *Settings → AI
   assistant* (needs no local model; the question and the retrieved passages
   are sent to that provider). It is never picked for you: with no local model
   loaded the page offers to load one right there, or you can choose the assistant.
8. **Copy it to another studio** — *Build the package → Studio-to-studio copy* writes an encrypted `.ftsrag` of the
   index that another Finetune Studio loads with *Import an index*. The package for your models is step 4.

**Use the Smart Brain for facts that change; use fine-tuning for style/response shape.**
Both read the same parse.

---

### 3. Test it (step **3 · test**)

The **Testing** page answers one question: *does it get MY material right?* With a built Smart Brain it opens on
**Smart Brain** mode:

1. **Write quiz** — the AI assistant writes questions from your indexed files (spread over every document), plus a few
   the files do not answer, where declining is the right reply. Items whose answer is not really in their passage are
   dropped, and out-of-date or duplicate documents are left out. Read a few questions, the unanswerable ones above
   all, before you trust the score.
2. Pick the **model under test** (the chat model that reads the passages, e.g. a 4B GGUF) and press **Run test**: the
   run saves every raw answer, nothing is scored by string matching.
3. **Judge** — an AI judge (any connection; by default the AI assistant's *Test judge* job) or you decide pass /
   partial / fail for each answer. *Judge automatically after each run* starts it by itself. Spot-check 10–20 answers
   with your own eyes — the judge is a model, not truth — and press 1/2/3 in the review pane to override a verdict.

You can also import your own quiz (JSON / JSONL with `q` and `expect`, `"expect_abstain": true` for questions the
documents do not answer). For a trained model, **Project quiz** and **Dataset check** test the training data instead.

**Benchmarks** (per project, optional) is separate: public exams — the official MMLU / GSM8K /
HellaSwag test sets from Hugging Face, or offline synthetic look-alikes — and a **Compare two runs**
Δ table with a per-question McNemar test. Useful, and not a substitute for the project test.

### 4. Package it (step **4 · package**)

The **Package** page lists each step from files to package with its state (files read, knowledge check, identity,
Smart Brain up to date, last test). **Build the package** opens the export form:

1. **Documents in the package** — *All documents*, or only some folders, tags or single documents ("8 of 15 documents
   go in the package"). Build a public chatbot from `public/` and an internal assistant from everything: what is left
   out is not inside at all.
2. **Search layers** — meaning search, word search, rank fusion, reranker (needs *Include models*: the embedder and
   reranker ship inside for offline meaning search, about 0.2 GB with MiniLM, about 4.5 GB with e5-large and
   bge-reranker-v2-m3) and answer generation
   (the package writes the answer with an OpenAI / Claude / OpenRouter / local connection; the key is set where it runs).
3. **Include original files** (off by default; anyone who can reach the server can download them) and **Hide file
   names in answers** for a public bot.
4. **Passphrase** — leave it blank and one is generated and **shown once**: copy it. The package is encrypted
   (AES-256-GCM); without the passphrase nothing in it can be read.
5. **⬇ Build & download standalone server**. Before it is handed over, the package opens itself with its own server
   and runs every search it ships (and, for a partial package, checks nothing outside it can be found).

Run it on any Linux or macOS machine with Python 3:

```bash
tar -xzf my-project-smart-brain-<id>-v1.0.0.tar.gz && cd rag-pack
echo 'the passphrase' > ../pass.txt && chmod 600 ../pass.txt
bash install.sh                     # a private venv with numpy + cryptography
bash run-http.sh --keyfile ../pass.txt     # http://127.0.0.1:8899 — chat page at /, MCP at POST /mcp
```

`bash setup.sh` does the same guided (port, start now, optional service, a copy-paste MCP entry for Claude Desktop,
Cursor, OpenClaw and other MCP clients; `bash run-mcp.sh` serves MCP on stdio). Agents use `rag_search`, `rag_info`,
`rag_profile`, and with originals `rag_list_documents` / `rag_get_document`.

**New versions.** Change files, update the Smart Brain, and export with **Package → New version of …**: the package
keeps its id, its version goes 1.0.0 → 1.1.0, and `CHANGES.md` (and the Package page's *Package versions*) list the
documents added, changed and removed. The new version keeps the package's documents, original files, hidden names and
audience unless you change them (anything wider asks you to confirm), and needs the same passphrase so running copies
keep working. Update a running copy in place with `bash update.sh NEW.tar.gz --keyfile ../pass.txt` (it keeps your
keyfile and the `rag.config.json` values you changed, and refuses another package, an older version, or a version that
opens up more than the running one unless you add `--allow-widen`). To edit a package someone
gave you, **Smart Brain → Import an index or open a package** turns it back into a project, original files included.

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

### 8. Export and use the trained model

- **Export** (the Package page's *Trained models* section): **merged** (adapter onto a 16-bit
  base — auto-merged before GGUF; failures land on the run row, never
  silent), **GGUF** (q4_K_M … q8_0, needs llama.cpp tools on the host; tick **importance matrix**
  for better low-bit quality, measured on the run's own training data), **abliterated**
  (refusal-direction edit of a merged checkpoint; offered, not yet run end to end on a real model), and
  **Heretic** (an optional separate tool, installed from *Settings → Heretic*, AGPL-3.0, that searches the
  refusal-removal parameters to keep the model close to its original behaviour).
- **Pin a version** — Quick work's Step 6 pins the exact dataset + corpus +
  run + base model into an immutable project version, so any result is
  reproducible and branchable.
- **Models** page: every export in one table — click a row to expand the
  parent run, settings, and directory contents; ▶ Open in inference.
- **Chat**: load a safetensors **or GGUF** model, stream, image input for
  multimodal, per-session temperature/top-p/system prompt. History
  persists; idle models auto-unload. A trained model also makes a good reader for the Smart Brain: test it on the
  Testing page in Smart Brain mode.

## Common first-run problems

| Symptom | It means | Fix |
|---|---|---|
| Smart Brain says "Needs an update" though you changed nothing | The Vision reader read pictures after the build | Click **Update Smart Brain**: the readings go in |
| Package page: Build is off | Superseded documents block packaging | Open the knowledge check from the line above the button; remove or accept them |
| Package searches by keyword only where it runs | No embedder inside and no embeddings endpoint | Tick **Include models**, or give an embeddings endpoint (the line above Build says which) |
| `update.sh` refuses: "opens up more than the installed" | The new version carries more documents, originals or shown names | Check it is meant for this copy, then add `--allow-widen` |
| `update.sh` refuses: "give --keyfile" | The new version is encrypted | `--keyfile /path/to/passphrase.txt` (or put it in `passphrase.txt` beside `server.py`) |
| Testing has no model to choose | No export and no local model | Pick an **AI assistant connection** in a Smart Brain test (no GPU needed) |
| "No datasets yet" in Training (optional) | Export step didn't run/complete | **pairs** → Export approved → Training; look for the green success + CTA |
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
- [KNOWLEDGE-PACKAGE-DEPLOYMENT.md](KNOWLEDGE-PACKAGE-DEPLOYMENT.md) — connect local agents, set up trusted HTTPS access, or build a public bot gateway
- [INSTALL.md](INSTALL.md) — install and troubleshooting
- [DEPENDENCIES.md](DEPENDENCIES.md) — every dependency and why it's there
