---
id: export
title: Package & export page (step 4)
page: export
keywords: package export gguf build public bot internal assistant scope originals version mcp quant q4_k_m merged abliterated heretic llama.cpp ollama imatrix
---
## Purpose
Step 4 of a project: build the **knowledge package** your models use. One encrypted archive holds the Smart Brain index of your documents, the identity (who it speaks for, how to answer), a small Python server that speaks HTTP and MCP, an installer and run/update scripts. The page first says what is left to do (the readiness checklist), then holds the package form, then the package versions. Training exports (GGUF, merged, abliterated, Heretic) are further down, under *Trained models*, for the optional training path.

**Trained models (optional training path).** Take a finished training run out of the studio: GGUF for llama.cpp, Ollama and LM Studio; Merged (full safetensors); Abliterated; or Heretic (the optional tool that removes refusals with a parameter search). Adapter-only runs are merged onto their base automatically at export time.

1. Select a finished run in **Training runs**.
2. Tick formats: GGUF with q4_k_m is pre-ticked (recommended). Tick more quants if wanted (f16, q8_0, q6_k, q5_k_m, q4_k_m, q3_k_m, q2_k).
3. **Export selected** (about 20–40 s for a small model). Files appear under *Trained exports* and become selectable in Chat and Testing.
4. **Overwrite existing** re-exports over old files.
5. **Importance matrix (imatrix)**: before quantizing, the app measures which weights the run's own training data relies on (llama-imatrix, up to 128 chunks of 512 tokens) and keeps those more precise. Same file size, better q4/q3/q2 quality, a few minutes extra. The files land in the run's `gguf-imatrix/` folder beside the plain `gguf/` ones; f16 and q8_0 are written without it, so tick at least one smaller quant.

## Workflow
1. Read the checklist (`#package-readiness`): files read, knowledge check, identity, Smart Brain up to date, a test, the last package. Each step links to where it is fixed. Superseded documents nobody accepted stop the build.
2. **Who it is for** (`#mcp-name`, `#mcp-audience`): name the package ("<project> — public/" by default) so two packages of one project are told apart; *Customers (public bot)* also hides file names and leaves the original files out.
3. **Documents in the package** (`#mcp-scope`): all, or only some folders, tags or documents (a public bot gets only `public/`).
4. **Files and sources**: include the original files (agents and the chat page can download them) and/or hide file names in answers.
5. **Answers**: off returns passages with sources (your own model writes the answer); on, the package also answers with an AI connection (the key is never inside).
6. **Version**: a new package (1.0.0, new id) or *New version of …* an earlier one (same id, lists what changed).
7. **Build & download the package** (`#mcp-pkg-btn`). Copy the passphrase when it is shown: it is not stored. The package opens and searches itself before the download starts (*Self-test passed*).
8. Run it: `bash install.sh`, then `bash run-http.sh` (chat page + `/mcp`) or `server.py --mcp` for local MCP clients; update a running copy with `bash update.sh NEW.tar.gz`.

## Key controls
- `#package-readiness` — what is left before packaging, each step with its state (Done / Review / Fix this / To do).
- `#package-form` — the package form.
- `#mcp-name` — the package's name (Package versions and the archive's file name).
- `#mcp-audience` — used by your team or by customers (a label; Customers also picks the public-bot options).
- `#mcp-pkg-btn` — build, self-test and download the package.
- `#mcp-layers` — the search layers, one switch each.
- `#mcp-sem` — export meaning search (vectors).
- `#mcp-kw` — export word search (BM25).
- `#mcp-rerank` — export the reranker (needs Include models).
- `#mcp-answer` — answer generation (the "G" in RAG) with an LLM endpoint.
- `#mcp-llm-preset` — fills the answer endpoint and API kind for OpenAI, Claude, OpenRouter or a local server.
- `#mcp-pipeline` — how a question travels through the package.
- `#mcp-scope` — Documents in the package: **All documents** (default) or **Only these** — folders (with their document counts), file tags and single documents. A document goes in when it matches any choice. The package then holds only those documents' passages, vectors, a word index recomputed over them and their files: nothing else can be found or read through it. Use it for a public chatbot from the `public/` folder next to an internal assistant with everything. CLI: `fts brain export --include public/ [--exclude PREFIX] [--tag TAG]`.
- `#mcp-scope-summary` — how many of the index's documents go in ("12 of 80 documents"); the package README and `rag_info` state the same scope, and the self-test searches for words of left-out documents and expects nothing.
- `#mcp-search-where` — above the Build button: how the package finds passages where it runs. A package without the models inside ships the vectors but no embedder, so it searches by exact words only until the host sets an embeddings endpoint; the line says so in amber. Fix it with **Include models** (about 0.2 GB with MiniLM, 4.5 GB with e5-large) or `#mcp-embed-url` (under Advanced): an OpenAI-compatible `/embeddings` endpoint serving the index's embedding model where the package runs (its key is `RAG_EMBED_API_KEY` there, never in the package; `#mcp-embed-model` names the model). The self-test lends the studio's embedder and its result says the same ("where it runs: word search only …"). CLI: `fts brain export --embed-base-url URL --embed-model NAME`.
- `#package-versions` — package names. The archive and the folder it extracts to are `<project>-<package>-v<version>-<YYYYMMDD>` (a second package with the same name the same day gets its short id added), so two exports never share a file or a folder. **Installs as** is the stable install name `<package>-<id8>`: the same for every version of one package, different for every package. `setup.sh --service` names the systemd unit `fts-brain-<install name>` and `MCP-ENTRY.txt` the MCP server the same, so two packages on one host never collide and an update replaces the same install. A copy installed before 0.6.0 (folder `rag-pack/`, unit `fts-brain-rag-pack`) updates in place as before; `bash update.sh --rename` (or `setup.sh --rename`) then moves its service and MCP entry to the stable name. The running package reports its name, version, build date, id, studio version, scope, layers and document count on `GET /version` (no token needed, no document names), in `/health`, the `X-Package-Version` response header, MCP `serverInfo` (name = install name, title = package name, version), the MCP instructions, `rag_info`, the chat page footer and the README's first line (`package.json`); the package check compares `/version` with `package.json`.
- `#mcp-versioning` — the package's version. **A new package** gets a new id at 1.0.0; **New version of …** (`#mcp-base`, from this project's package history) keeps that package's id, bumps the minor version (`#mcp-version` overrides it) and lists the documents added, changed and removed since then (`CHANGES.md` in the package, the Package page). `#mcp-base-upload` reads an exported archive (+ passphrase) as the base when it is not in the history. Every package carries its id, version, date, parent version and document list (name, sha256, size) in `rag_info` / `GET /health`. Picking a base loads its settings into the form: its documents (scope), original files, hidden file names and audience; a new version keeps them unless you change them, and anything wider (documents outside its scope, originals turned on, file names shown, a public audience changed, encryption off) asks for a confirmation, is listed in `CHANGES.md` and is refused by `update.sh` on a running copy unless `--allow-widen`. An encrypted new version needs the current version's passphrase (so running copies keep working); `#mcp-newpass` gives it a new one on purpose. After a build the form goes back to **A new package**. A running copy updates in place with `bash update.sh NEW.tar.gz --keyfile passphrase.txt` (the keyfile defaults to `passphrase.txt` beside `server.py`; keeps the `rag.config.json` values you changed and takes the new version's other settings; refuses another package id unless `--force`). CLI: `fts brain export --new-version-of ID` / `--base ARCHIVE` / `--version X.Y.Z` / `--allow-widen` / `--new-passphrase`.
- `#mcp-hide-names` — Hide file names in answers (public bot; off by default): the package's answers cite no source files (`[Harrowgate_deal_sheet.xlsx]` would look internal to a customer). The answer model sees passages by number only, any file name it writes anyway is removed, and the package's chat page shows no source list. Your own tools still get the sources by asking: `POST /ask` with `"sources": true`, `rag_answer` with `include_sources`. `/search` is unchanged. CLI: `fts brain export --hide-file-names`; where it runs: `RAG_HIDE_FILE_NAMES=1`.
- `#mcp-originals` — Include original files: the uploaded files themselves ship in the package (encrypted with the corpus). Agents fetch one with the MCP tool `rag_get_document` (embedded file, up to 20 MB inline; larger ones come back as a link), `rag_list_documents` answers "which files do you have about X", browsers download at `GET /files/<id>`, and the package's chat page shows a download link under each source. Anyone with access to the server can download them: leave it off for a public bot. Off by default. CLI: `fts brain export --originals`.
- `#package-versions` — the project's packages, one block per package: its name, who uses it, scope, document count and id, then each version (date, documents, original files in or out, file names hidden or shown, what changed since the parent) and the latest version's change report. *New version of this* selects it in the form.
- `#export-runs` — training runs to choose from.
- `#gguf-quants` — quantization check boxes.
- `#gguf-imatrix` — quantize with an importance matrix measured on the training data.
- `#export-btn` — start the export (disabled until a run is selected).
- `#export-force` — overwrite existing exports.
- `#trained-exports-table` — finished exports.

## What to check
The self-test line after a build says *Self-test passed*; the versions list shows the new version with the scope you chose. For trained models: two GGUF files for two quants under the run's `gguf/` folder. Prefer q6_k or q4_k_m for small models. Q8_0 once crashed the whole service on some models via a native llama.cpp bug; the loader now caps the micro-batch at 256, but there is no reason to pick the riskiest quant for a 0.6B model.

## Common mistakes
Exporting GGUF when it is marked unavailable (the page shows why); ticking the importance matrix for a run whose training file was deleted (refused with the reason); ticking Abliterated without understanding it removes refusal behaviour.

Terminal alternative: `fts export start <run-id> --project <project-id>` queues the same export job; `fts export list`, `status`, and `cancel` inspect or stop it. Run `fts export start --help` for quant, imatrix, merge, and overwrite options.
Ticking Heretic when it is greyed out (Settings → Heretic installs it; the page says why it cannot run). Heretic needs free VRAM for the model plus about a tenth and 1 GB: with less, the export stops at once, saying how much is needed and which programs hold the card. If another program takes the VRAM during the search, the export pauses and says so, then resumes by itself from the finished trials once there is room; after 30 minutes it stops and the next export of that run continues from those trials (a cancelled search continues the same way). Heretic also needs more GPU time than Abliterated: it searches parameters across many trials instead of applying one fixed strength.
