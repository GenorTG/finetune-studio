# ⚡ Finetune Studio

**Train, run, and evaluate large language models — entirely on your own hardware.**

A self-hosted workshop that gives you full control over the model training
lifecycle: build RAG corpora, fine-tune with LoRA, chat with local models,
run evaluation suites, compare runs side-by-side — all from one dark-themed
WebUI that lives in your browser. The UI, the data pipeline, and the test
suite all live in one repo.

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: PolyForm Noncommercial](https://img.shields.io/badge/license-PolyForm%20Noncommercial%201.0.0-ff8800.svg)](LICENSE)
[![GPU: CUDA](https://img.shields.io/badge/GPU-CUDA-76b900.svg)](#installation)
[![WebUI](https://img.shields.io/badge/WebUI-FastAPI%20%2B%20Jinja2-009688.svg)](#how-it-works)
[![Presentation page](https://img.shields.io/badge/%E2%9A%A1_presentation_page-live-00ff66.svg)](https://genortg.github.io/finetune-studio/)

---

## Table of contents

- [Why I built this](#why-i-built-this)
- [Who it's for](#who-its-for)
- [The four use cases](#the-four-use-cases)
- [What's new in the early-beta line](#whats-new-in-the-early-beta-line)
- [Why this over doing it manually?](#why-this-over-doing-it-manually)
- [Walkthrough — the pages](#walkthrough--the-pages)
- [Quickstart](#quickstart)
- [Installation](#installation)
- [How it works](#how-it-works)
- [Tech stack](#tech-stack)
- [Documentation](#documentation)
- [Contributing](#contributing)
- [License & attributions](#license--attributions)

> **New here?** The fastest path from zero to a trained model is the
> step-by-step [**Tutorial**](docs/TUTORIAL.md) — or open **⚡ Quick work**
> inside a project and chain files → pairs → train → test in one page.
> Common first-run pitfalls are covered in both.

---

## Why I built this

Training a language model today involves juggling **dozens of moving pieces**:
managing CUDA dependencies, downloading and quantizing weights, building
chunking pipelines for RAG, writing training loops, babysitting VRAM,
finding a benchmark suite, monitoring loss curves, packaging the result,
serving it, comparing it against the base model…

There are great point-solutions for each piece (HuggingFace TRL, LM Studio,
PortableRAG, llama.cpp, MMLU…) but **no single workshop that wraps the whole
lifecycle in one coherent interface**. You end up with five tools open,
three terminal windows, and a Notion page of "things to remember."

Finetune Studio exists to fix that. It is a **browser-based control room**
for everything between *raw documents* and *a chat-ready, evaluated,
fine-tuned model* — running entirely on your hardware, with no cloud
dependency.

The goal isn't to compete with industrial training platforms. It's to give
independent researchers, hackers, hobbyists, small teams, and educators a
**single place to do serious work** without the friction of cobbling it
together themselves.

---

## Who it's for

- **Independent researchers** prototyping custom models without paying cloud
  bills.
- **Hobbyists** who want to fine-tune Llama 3 on their RTX 3090 without
  reading ten blog posts first.
- **Small teams** that need a private inference + RAG server and don't want
  to ship sensitive data to a third party.
- **Educators** teaching an LLM workshop — one app, all the steps visible.
- **Anyone** who wants a polished local alternative to closed hosted
  services.

You don't need all the use cases. Most users start with one (RAG, or
fine-tuning, or just chatting) and grow from there.

---

## The four use cases

You don't have to use every feature. Pick what you need.

### Use case 1 — Build a RAG on private documents

> *"I have a pile of PDFs and I want to ask questions about them in plain
> English — without uploading anything to the cloud."*

The **RAG** workflow takes a folder of files, chunks them, embeds them
into a vector store, and exposes them for similarity search. The studio
handles:

- File ingestion (PDF, DOCX, TXT, MD, HTML, code, CSV, JSON, images via
  OCR)
- Configurable chunking strategy
- Embedder model selection (any SentenceTransformer-compatible model)
- Live build progress via Server-Sent Events
- A docs-indexed panel showing every indexed chunk with a per-doc rebuild
  button + view-chunks modal
- **Standalone export**: one download turns the corpus into a self-installing
  package (its own venv) that serves search over **MCP**
  (Claude Desktop / OpenClaw / Cursor) *and* a plain **HTTP API** —
  `bash setup.sh` does the rest: install, port choice, start-now, optional
  persistent service (`--uninstall` removes it), and a copy-paste MCP entry
  with real absolute paths. Keyword
  search works offline out of the box; point it at any OpenAI-compatible
  embeddings endpoint for semantic search — or tick "Include models" and the
  embedding model + reranker ship **inside** the package: full offline
  semantic search with no external service at all (~2.3 GB archive)

📸 **See:** the RAG page with the live corpus inventory.

### Use case 2 — Fine-tune a model on your data

> *"I want to teach a small open model how my company talks, how to answer
> support questions, how to follow our style guide."*

The **Training** workflow lets you point the studio at a JSONL of
prompt/completion pairs (or your own dataset), pick a base model, and run a
supervised fine-tune (SFT) with LoRA or QLoRA. The studio handles:

- Adapter merging, model export, VRAM profile checkpoints
- Real-time loss + step counter, gradient norm, learning rate
- Configurable LoRA rank, target modules, batch size
- A merge step that produces a self-contained checkpoint you can load in
  chat
- Per past-run Actions (⭐ Set Production, ▶ Inference, ⬇ Download) so
  you can promote / inspect / pull down any of your previous runs

📸 **See:** the Training page with the live CpuChip sprite that pulses as
training runs.

### Use case 3 — Benchmark before you ship

> *"I trained two variants. Which one is actually better?"*

The **Benchmarks** workflow runs offline evaluation suites (built-in
MMLU-/GSM8K-/HellaSwag-style smoke fixtures, plus project-local suites) and
gives you a **comparison table**. Two tabs: **Recent scores** (rolling
history) and **Compare two runs** (pick any two trained exports, get the
per-suite Δ table).

These are local smoke suites styled after well-known benchmarks — not the
full HuggingFace dataset downloads — so they stay offline and fast.

📸 **See:** the Benchmarks page with the live score bars and the compare
results panel.

### Use case 4 — Just chat with local models

> *"I want to run Llama 3 locally with my own chat history, image support,
> and a clean UI."*

The **Chat** page is a polished client. Pick a safetensors or GGUF model on
disk, load it (with auto-unload on idle to free VRAM), and chat. Image
input for multimodal models. Per-session temperature / top-p / system
prompt. Conversation persists across reloads.

📸 **See:** the Chat page with the RobotHead sprite that pulses its mouth
as tokens stream in.

---

## What's new in the early-beta line

Since v2 the app grew a spine — and a lot of muscle:

- **⚡ Quick work** — the entire model pipeline on one page (upload →
  pairs → dataset → train → test → pin), a live status pill per step, and
  **Run steps 2–5** to chain dataset → training → test unattended.
- **Two-card flow picker on the project home** — 🧠 Train a model vs
  📚 Search my files (RAG) — with numbered, flow-scoped sub-nav
  (`1 files → 2 pairs → 3 train → 4 test → 5 use it`).
- **File browser v2** — folder chips with drag-to-move, tag pills, bulk
  actions, zip export, per-file usage chain (file → pairs → datasets →
  runs → RAG), parsed-content search, thumbnails, columns picker, 7-day
  trash with restore-all.
- **100 % data guarantee** — every dataset export fills un-mined chunks
  with verbatim extractive pairs; a dataset can never ship with silent
  holes.
- **Full-coverage test suites** — auto-suites test every dataset row
  (N rows → N questions); sampling is an explicit, labeled opt-in.
- **Project versions** — immutable manifests pinning datasets + sources +
  corpora + runs + base model, with lineage; pin from Quick work Step 6.
- **Standalone RAG package** — the corpus ships as a self-installing
  tarball (MCP + HTTP, `setup.sh`, optional bundled embedder + reranker
  for fully offline semantic search).
- **Parsed-text editor** — fix a bad parse in-app; corrections feed
  mining and the next RAG build while raw bytes stay immutable.
- **Per-commit build chip + self-update** — the header pins the exact
  deployed version (`v0.1.0.N`); Settings → Apply update pulls, repairs,
  migrates and restarts.
- **Dual-theme visual QA** — every route probed at 1920/1440/1270/768/375
  in dark and light: no clipped actions, no sub-11px text, WCAG-checked
  contrast, both themes.

---

## Why this over doing it manually?

| What you'd do manually | What Finetune Studio does |
|---|---|
| Write a Python script to download weights from HF | Browse HF Explorer in-app, click download |
| Configure transformers + accelerator + trl + peft + bitsandbytes | Pick base model + LoRA rank + batch size in a form |
| Write your own chunking + embedding loop | Click "Build" — chunking, embedding, indexing all wired |
| Open 4 terminal windows to monitor training | Watch the live progress bar in one panel |
| Hand-craft an eval script from scratch | Pick a suite, click Run, get a score table |
| Open a second tab to compare two trained runs | Pick run A + run B, click Run comparison, get the diff |
| Lather / rinse / repeat for every file rename | Click rename, type new name, ⏎ |
| Lose an afternoon to CUDA install issues | One `./install.sh` does it |
| Re-discover package version conflicts every 3 months | Pinned dependencies, versioned upgrades |
| Debug a model load failure from "ENOMEM" | Actionable error: how much VRAM is needed vs free, and which other GPU processes are holding it |

The point isn't that any single piece is impossible. The point is that
**all of it is now in one place**, with a shared project concept that ties
it together, and all your data lives in `~/.finetune-studio/` where you
can find it.

---

## Walkthrough — the pages

### Dashboard (`/`)

The home view. Host resources (RAM / VRAM bars updated live), your
projects, quick-create buttons.

![Dashboard](docs/screenshots/01_dashboard.png)

### Project home (`/projects/{pid}`)

Answers “where do I start?” before you touch any tab: two flow cards —
🧠 **Train a model** (what it is, why it's slow, 5 ordered steps) and
📚 **Search my files / RAG** (minutes, no training, 4 steps) — plus live
counts for files, datasets, runs and models, recent training history,
latest exports, and the activity feed.

![Project home](docs/screenshots/02_project.png)

### ⚡ Quick work (`/projects/{pid}/work`)

The whole pipeline on one page: six numbered cards — **1** upload files,
**1b** build a RAG index (one click, live coverage pill), **2** make
question-answer pairs, **3** build the dataset, **4** train, **5** test,
**6** pin & use — each with its own button and status pill, plus
**Run steps 2–5 now** to chain dataset → training → test unattended while
progress streams below. This is the recommended path; the individual pages
are there when you want fine control.

![Quick work](docs/screenshots/03_quick_work.png)

### Files — step 1 of both flows (`/projects/{pid}/data`)

The file browser: drag-drop upload (SHA-256 dedup, MIME auto-folders, raw
bytes immutable), renameable folder chips with drag-to-move, tag pills with
click-to-filter, bulk actions (delete / restore / move / re-parse / tag),
thumbnails, sortable + pickable columns, pagination, parsed-content search,
⬇ zip export, ℹ️ a per-file usage chain (file → sources → pairs → datasets →
runs → RAG), a built-in **parsed-text editor** (corrections feed mining and
the next RAG build without touching the original), and a 7-day soft-delete
trash with restore-all.

![Files](docs/screenshots/04_files.png)

### Pairs — step 2 (`/projects/{pid}/data-prep`)

Question-answer pairs: mine training pairs from your parsed sources with an
extractive prep job (no LLM), the Chat Agent, or manual JSONL upload; triage
with bulk approve / reject. **Export approved → Training** writes a ShareGPT
JSONL dataset — and the coverage gate fills any chunk mining missed with
verbatim extractive pairs, so a dataset can never ship with silent holes:
**100 % of your material is always in the training data.**

![Pairs](docs/screenshots/05_pairs.png)

### RAG — the search-my-files workspace (`/projects/{pid}/rag`)

Build an index, search it, tune retrieval parameters, and take it with you.
Sections are numbered in workflow order: **1. Index status**, **2. Build the
index** (⚡ Quick index promotes every parsed file in one click), **3. Search
settings**, **4. Documents in the index**, **5. Search test**, **6. Ask the
model (grounded)**, **7. Download / export the corpus**, **8. Shared search
models**. The export card offers both the plain corpus archive and the
**standalone package** (`GET /api/projects/{pid}/rag/mcp-package`) — a
self-installing tarball with the index, a single-file Python server,
`setup.sh`, MCP config example, and a README; it speaks MCP over stdio and
REST over HTTP, and can ship the embedder + reranker inside for fully
offline semantic search. Index is persisted on disk as
`~/.finetune-studio/rag_corpora/{pid}/{chunks.parquet,vectors.npy,
bm25.json,sources/*.txt}`. The CorpusNode sprite shows document nodes
connecting to embedding vectors in real time.

![RAG](docs/screenshots/06_rag.png)

### Training — step 3 (`/projects/{pid}/training`)

Configure a fine-tune, watch loss curves live. Base-model and preset
pickers recommend rank/LR/epochs from your base size and dataset size. Each
past run in the **Past runs** table carries Actions (⭐ Set Production,
▶ Inference, ⬇ Download) so you can promote / inspect / pull down any of
your previous runs without re-opening individual pages. The CpuChip sprite
flickers as the GPU accelerates.

![Training](docs/screenshots/07_training.png)

### Testing — step 4 (`/projects/{pid}/testing`)

Did it learn *your* material? Auto-generated project suites test **every
row** of the training dataset (N rows → N questions; sampling is an
explicit, labeled opt-in), strict-substring judged and reviewable
case-by-case.

![Testing](docs/screenshots/10_testing.png)

### Models — step 5 (`/projects/{pid}/models`)

Every trained export from every run, in a 7-column table (Name / Format /
Size / Source run / Created / Copy path / Actions). Click any row's name
to **expand** it and see the parent training run, the training settings
(LR, rank, batch, epochs, max seq length, merge-on-save), and a top-level
directory listing. ▶ Open in inference jumps straight to chat.

![Models](docs/screenshots/08_models.png)

### Export (`/projects/{pid}/export`)

Pick a run, pick a format, hit RUN. Supported formats today:

- **merged** — adapter merged onto a compatible base (safetensors)
- **abliterated** — refusal-direction edit of a merged checkpoint
- **GGUF** — when llama.cpp conversion tools are installed on the host;
  otherwise the API fails with a clear install hint
- **GPTQ** — when `auto-gptq` is installed; otherwise fails honestly

The trained-exports table uses the same expand-row pattern as Models —
click a row to inspect its contents, then ▶ Open in inference.

![Export](docs/screenshots/09_export.png)

### Chat (`/projects/{pid}/chat`)

Pick a safetensors or GGUF model, load (auto-unload on idle frees VRAM),
and chat — streaming with sprite feedback, image input for multimodal
models, per-session temperature / top-p / system prompt, persistent
history, RAG-attach for grounded answers, and Agent mode for tool-driven
dataset curation.

![Chat](docs/screenshots/13_chat.png)

### Benchmarks (optional, `/benchmarks`)

Public exams, separate from your project test: offline smoke suites
(MMLU-/GSM8K-/HellaSwag-style) plus project-local suites, real HF splits
when you allow downloads. Two tabs: **Recent scores** (live history) and
**Compare two runs** (pickers + a per-suite Δ table). The BenchBars sprite
fills in as scores arrive.

![Benchmarks](docs/screenshots/11_benchmarks.png)

### Settings (`/settings`, per-project under `/projects/{pid}/settings`)

Per-project settings including a **WebUI log tail** card (Refresh +
Auto-refresh) for debugging long-running jobs without leaving the browser.
Also hosts the in-app **Apply update** controls for the self-update
pipeline and the **Replay Tutorial** button for the 8-step onboarding tour.

![Settings](docs/screenshots/12_settings.png)

---

The session bar at the top of every page is a Tmux-style tab strip with
three groups: **[SYS]**, the active **[project]**, **[TOOLS]**. Inside a
project, a numbered step strip shows the flow you're in —
`1 files → 2 pairs → 3 train → 4 test → 5 use it` for training,
`1 files → 2 build → 3 search → 4 chat` for RAG. The active tab has a
phosphor-green notch that punches into the active-path bar below it.
`Ctrl+K` opens the command palette — fuzzy-search any page in any
project.

---

## Quickstart

```bash
git clone https://github.com/GenorTG/finetune-studio.git
cd finetune-studio
./install.sh                       # auto-detects GPU
source .venv/bin/activate
fts web --host 0.0.0.0 --port 7860
```

Open http://localhost:7860. The first visit walks you through an 8-step
onboarding tour.

---

## Installation

See **[docs/INSTALL.md](docs/INSTALL.md)** for full step-by-step
instructions covering Linux (Ubuntu / Fedora / Arch), Windows 10/11 (WSL2),
and macOS (Intel + Apple Silicon), with prerequisites and how to install
each one from its official source.

---

## How it works

```
┌───────────────────── BROWSER (any modern) ─────────────────────┐
│                                                                │
│   Tmux-style session bar ─── project tabs ── status pill       │
│   ┌──────────────────────────────┐  ┌────────────────────────┐ │
│   │ FastAPI + Jinja2 templates   │  │ Live SSE for progress  │ │
│   │ Vanilla JS / CSS (no build)  │  │ Chat streaming         │ │
│   │ Playwright E2E smoke suite   │  │ Tail /api/…/logs       │ │
│   └──────────────────────────────┘  └────────────────────────┘ │
└────────────────────────────────────────────────────────────────┘
                              │  HTTP + Server-Sent Events
                              ▼
┌───────────────────── PYTHON SERVER (local) ─────────────────────┐
│                                                                │
│   FastAPI app ── projects API ── RAG pipeline                  │
│               ── training runner (trl+peft)                    │
│               ── inference (transformers and/or llama.cpp)     │
│               ── benchmarks (offline smoke + local suites)     │
│                                                                │
│   Storage: SQLite for project metadata                         │
│            + ~/.cache/huggingface for model weights             │
│            + ~/.finetune-studio/rag_corpora/{pid}/  (parquet)   │
└────────────────────────────────────────────────────────────────┘
```

- **Backend:** FastAPI + Jinja2. ASGI server (Uvicorn).
- **Frontend:** Vanilla HTML + CSS + JS. No build step. Sprites are
  inline SVG with CSS keyframe animations.
- **Storage:** SQLite for project metadata, PortableRAG corpus (parquet +
  per-project sources dir) for RAG vectors, HuggingFace `~/.cache/`
  for model weights.
- **Streaming:** Server-Sent Events for live progress on RAG build,
  training, activity, and related long-running jobs; project log tail
  for on-demand server logs.
- **Inference:** HuggingFace Transformers for safetensors / LoRA merges;
  llama.cpp (via `llama-cpp-python`) for GGUF when available.
- **Export:** merged and abliterated are first-class; GGUF needs llama.cpp
  conversion tools on the host; GPTQ needs `auto-gptq`. Missing tools
  return clear errors instead of fake success.
- **Testing:** Playwright live-browser smoke in `tests/e2e_ui_qa.py`
  (override target with `FTS_BASE`).
- **Updates:** self-healing pipeline — `./update.sh` or Settings → Apply
  update (pull → venv repair → dep sync → migrations → restart), with
  startup reconciliation of interrupted runs.

---

## Tech stack

Python — `fastapi` `uvicorn` `jinja2` `sqlite3` (stdlib) `pydantic`
ML — `torch` `transformers` `trl` `peft` `bitsandbytes`
RAG — `sentence-transformers` `PortableRAG` (parquet + sources) `llama-cpp-python`
Web — vanilla HTML/CSS/JS, Google Fonts (JetBrains Mono, Share Tech Mono,
VT323, Press Start 2P)
Tests — `playwright` `pytest`

Full list with versions and licenses → **[docs/ATTRIBUTIONS.md](docs/ATTRIBUTIONS.md)**

---

## Documentation

- **[docs/README.md](docs/README.md)** — doc map (start here)
- **[docs/TUTORIAL.md](docs/TUTORIAL.md)** — end-to-end first-model tutorial (users start here)
- **[HANDOFF.md](HANDOFF.md)** — current ops state / next steps for agents
- **[docs/PRODUCT-BRIEF.md](docs/PRODUCT-BRIEF.md)** — product north star & quality bar
- **[AGENTS.md](AGENTS.md)** — how agents should work this repo
- **[docs/INSTALL.md](docs/INSTALL.md)** — install on Linux / macOS / Windows, prerequisites, troubleshooting
- **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** — layers, route map, data flow, disk layout
- **[docs/DEPENDENCIES.md](docs/DEPENDENCIES.md)** — every dependency mapped to its consumers
- **[docs/DEPLOYMENT.md](docs/DEPLOYMENT.md)** — service, update pipeline, generic ops notes
- **[docs/REFACTOR-SPEC.md](docs/REFACTOR-SPEC.md)** — the locked architecture decisions + roadmap
- **[docs/ATTRIBUTIONS.md](docs/ATTRIBUTIONS.md)** — every package, version, license
- **[⚡ Presentation page](https://genortg.github.io/finetune-studio/)** — the gallery version of this README
- **/settings** — live debug info on your running instance (version, GPU, packages, paths) **+ WebUI log tail card**
- Tools → Settings → **Replay Tutorial** — built-in onboarding, always available

---

## Contributing

PRs welcome. Open an issue first if the change is large — this codebase
prefers clear minimal patches over sweeping refactors. New features should
add a focused unit/regression test under `tests/`, and browser-visible
changes should extend the Playwright smoke suite when practical.

---

## License & attributions

**[PolyForm Noncommercial License 1.0.0](https://polyformproject.org/licenses/noncommercial/1.0.0)** plus Finetune Studio Additional Terms. See `LICENSE`. Personal, scientific, and non-commercial use is free. Commercial use and commercial model training require a separate paid license — see `LICENSE` (Additional Terms) and `docs/LEGAL.md`.

> Note: this is *source-available* software, not an OSI-approved open-source license, because it restricts commercial use.

Every dependency used is open source — see
**[docs/ATTRIBUTIONS.md](docs/ATTRIBUTIONS.md)** for the full list,
versions, and licenses. Thank you to the maintainers of PyTorch,
Transformers, PEFT, llama.cpp, PortableRAG, SentenceTransformers,
FastAPI, Jinja, and the rest of the open-source ecosystem this is built
on.

---

<div align="center">

Built by a hacker, for hackers. No cloud, no telemetry, no lock-in.

</div>
