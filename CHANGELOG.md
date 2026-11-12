# Changelog

All notable changes to Finetune Studio. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and versions follow [Semantic Versioning](https://semver.org/) (see [CONTRIBUTING.md](CONTRIBUTING.md#versions-and-releases)).
0.x releases are betas; 1.0.0 will be the first stable Linux release.

Every feat/fix/perf pull request into `dev` adds its line under **Unreleased**; `scripts/release.py prepare`
turns that section into the release's notes.

## [Unreleased]

- The dashboard puts its knowledge-package actions immediately under the welcome text, ahead of machine telemetry.
- The in-app Guide now names controls by their visible labels instead of exposing internal page-anchor IDs.
- Encrypted package READMEs now use the Smart Brain knowledge-package name without exposing the source project name.
- The project flow bar points to the Smart Brain build until its index exists, then moves on to testing its answers.
- Testing quiz names now stay human-readable after reload (for example, “Smart Brain quiz · 11 questions” instead of “local · smart-brain-quiz (11 cases)”).
- The Files page no longer shows the Data Prep animation in its filter card.
- The Files page opens upload on an empty project; choosing files or a folder starts right away, with visible transfer and parse progress.
- Training: a run that stopped or failed before saving a checkpoint no longer offers Set production, Open in inference
  or Export; the buttons are off and the row says why. A stopped run that did save an adapter stays usable.
- When an answer says part of it is missing, the retry searches the cited document(s) first before falling back to
  the whole knowledge base.
- Pages, styles and scripts are sent gzipped and versioned styles and scripts are cached by the browser, so a first
  visit moves about a fifth of the bytes and later pages load from cache (streams and downloads are never compressed).
- The Smart Brain quiz checks “unanswerable” questions against the index's top 10 hybrid-search passages, so answers
  assembled from more than one document are less likely to be mislabeled as missing.
- The host resources card (RAM and VRAM bars) is only on the dashboard, Training, Inference and My models, below the page
  title and smaller on phones; the knowledge pages open on their title and next step (the top bar keeps the VRAM pill).
- The knowledge package leads everywhere: the dashboard's quick actions are New project, Open <last project> and
  Package it (model download and local chat are linked below); New project starts on the Smart Brain template with
  the base model and system prompt folded under "Training (optional)"; the Files page points to the Smart Brain and
  the Package step; the first-run tour goes project → files → answer model → test → package and opens by itself only
  on the dashboard; the Guide suggests package questions first.
- The Package page now holds the package form: who it is for, which documents go in, original files and file names,
  answers, version, then Build & download; server and search tuning fold under Advanced. Before the first Smart Brain
  build it says to build it first, and a blocking knowledge-check finding is shown next to the button with a link. The
  Smart Brain page keeps only "Copy this index to another studio" (`.ftsrag`).
- Packages get a name and an audience (your team / customers; customers also hides file names and leaves the original
  files out). Package versions shows each package's name, audience, scope, document count and id, and per version
  the original files and file-name setting; "New version of this" picks it in the form.
- Settings → Apply update finds `uv` where it was installed (Linuxbrew, `~/.local/bin`, `~/.cargo/bin` or the path the
  installer recorded) instead of failing every dependency step, and an update whose dependency sync or build fails
  now stops before the restart with the failed steps as its reason instead of ending "Update complete".
- The Package page opens at once: its checklist loads after the page, and the knowledge check behind it is kept until a
  file or the index changes (a 120-file project took 4 s per visit, now 0.2 s). The status pills sit next to each step
  instead of stretching across it.

## [0.3.0] - 2026-11-12

The knowledge-package beta. A project now goes **files → Smart Brain → test → package**: upload your documents, let
the knowledge check find what is wrong with them, say who the answers speak for, test the answers on a quiz written
from your own files, and export one encrypted package that any model can use over MCP or HTTP. Training a LoRA
adapter stays in the app as the optional extra for tone and format. Linux only (NVIDIA GPU or CPU), as before.

### Highlights: the knowledge package

- **One main path.** Projects lead with `1 files → 2 Smart Brain → 3 test → 4 package`; training is an optional group at
  the end of the project bar. The project checklist, Next hints, Quick work, the dashboard, the first-run tour and the
  app's description all follow it.
- **Knowledge check** (Smart Brain page, `fts brain lint`): finds unreadable files, duplicates, out-of-date versions
  (also across file types, when file names carry years or dates, and older sheets kept inside a workbook), documents
  that disagree, and leaked passwords or keys. An amendment is linked to the agreement it amends instead of being
  called a conflict; a lone `…_DRAFT` is shown as a draft; form fields every document fills in, and figures in unlike
  units, are not reported as conflicts. A package stops on out-of-date documents nobody accepted, on the Package page
  and in `fts brain export` (`--accept-superseded` overrides).
- **Identity**: your company, its personality and how it should answer go with every answer and test and are packed,
  encrypted, into the export (MCP `rag_profile` and the server instructions, HTTP `/profile`).
- **Tests need no training data**: Testing writes a quiz from your indexed documents, including questions they do not
  answer, runs it and has an AI judge (or you) decide each answer (`fts brain quiz`). Out-of-date and duplicate
  documents are left out of the quiz.
- **The Package page** shows each step from files to package with its live state. Every package opens itself with its
  own server before it is handed over and runs every search it ships, fetches an original file when it has them,
  and for a partial package checks that nothing outside it can be found (`fts brain readiness`).
- **Packages are versioned**: a stable package id, a version, a build date and a document list (name, sha256, size)
  in `rag_info`, `/health`, the README and `package.json`. *New version of …* keeps the id, bumps the version and
  reports the documents added, changed and removed (`CHANGES.md`, the Package page's *Package versions*). *Open a
  package* (Smart Brain → Import, `fts brain import`) turns a package back into a project to edit, with its original
  files back in Files. A running copy updates in place with `bash update.sh NEW.tar.gz`, keeping its config and keyfile.
- **Audience scope**: a package can carry only some folders, tags or single documents ("8 of 15 documents";
  `fts brain export --include public/ --exclude … --tag …`). Build a public chatbot from `public/` and an internal
  assistant from everything; what is left out is not inside at all.
- **Original files on request** (*Include original files*, `--originals`; off by default, encrypted with the corpus):
  agents fetch a file with the MCP tool `rag_get_document` and list them with `rag_list_documents`, browsers download
  at `GET /files/<id>`, and the package's chat page links each source's file.
- **Public bots can hide file names** (*Hide file names in answers*, `--hide-file-names`, `RAG_HIDE_FILE_NAMES=1`).
- **Every way in**: packages speak MCP over stdio and over HTTP (`POST /mcp`, today's clients and the stateless
  2026-07-28 protocol; tools are marked read-only), plain HTTP, and a browser chat at `/` that answers through
  OpenAI, Anthropic Claude, OpenRouter or a local server (the key is set where it runs, never in the package).
  Each search layer (meaning search, word search, rank fusion, reranker, answer generation) is its own switch, and
  the package ships only the files the layers you keep need.
- **Better answers**: an answer that says part of the question is not stated searches once more for that part
  (*Search again for a missing part*); a question that goes through another item can ask for a follow-up search
  (Quillmoor benchmark, API model: 81.1 % → 96.1 % with the new reranker); one long document can no longer fill the
  top results (at most 3 passages per document first); new indexes on a GPU rerank with bge-reranker-v2-m3 (right
  passage in the top 5 for 89 % of questions instead of 79 %). Answers show their bold, code, headings and lists.
- **Folders nest**: upload a whole folder and its sub-folders are kept (`public/faq/returns.md`), in Files, the index,
  the package picker, manifests and opened packages.
- **Pictures are read on upload** by the vision reader when one is set (Settings → AI assistant → Reading pictures):
  chart images, table screenshots and scanned pages. Files and the knowledge check say when only OCR read them.

### Added

- **Heretic** (optional tool, Settings → Heretic or `fts heretic install`; AGPL-3.0, installed on demand): removes a
  model's refusals with a parameter search that keeps the model close to its original behaviour, as a new Export
  format beside Abliterated, Merged and GGUF. It keeps the trial with the fewest refusals inside a KL budget (0.1),
  shows the kept trial, its refusal rate and KL drift, pauses and resumes when another program takes the GPU, and
  resumes a cancelled search from its finished trials.
- **AI assistant** (Settings): one main assistant, local or any OpenAI-compatible API, does every helper job (mining,
  study sets, reworded quizzes, preference pairs, the test judge, drafting use-case examples, reading pictures, the
  Guide). Any number of API connections, each with its own encrypted key; any job can use its own connection.
  The Smart Brain chat can answer through it, as a deliberate choice (it starts on the local model).
- **The Guide can do the work**: start mining, build the dataset or the Smart Brain, train, test, pin a version;
  it asks before each action or runs hands-free. It never approves pairs, exports a model or deletes anything.
- **Training from a use case**: extraction, classification, house style or distillation examples (the assistant can
  draft them from raw texts for you to approve), checked rows, a held-out test and a comparison with the untrained base.
- **Importance matrix (imatrix)** for GGUF exports: better q4/q3/q2 quality at the same size.
- **Project versions**: each version lists what it holds, builds on an earlier one and shows its lineage.
- **CLI coverage of the web UI**: `fts projects`, `files`, `mining`, `qa`, `dataset build`, `brain` (`export`, `import`,
  `bundle`, `lint`, `quiz`, `readiness`, `pictures`), `export`, `usecases`, `assistant`, `compute-device`, `update`,
  `models`, `versions` (see `docs/CLI.md`).
- Buttons for things that only worked through the API: My models → Rescan folders, Smart Brain → Re-embed only and
  Stop build, Pairs → Remove near-duplicates and Data audit, Activity → Cancel download, Compare → delete a saved
  comparison, Model library → delete a downloaded model.

### Changed

- Installs, Repair and updates install the exact versions that were tested (`pins/`: Python packages, PyTorch,
  llama.cpp, Heretic's packages) and use uv's own Python 3.12. Repair keeps the old environment until the new one passes
  its health check.
- A calmer slate-and-teal look with clearer navigation, forms, tables and focus states; AA contrast in the light theme.
- The Smart Brain page leads with its state and the one next action, then the conversation; everything else is under
  Manage. The build bar shows passages embedded, time taken and time left; a build can be stopped and never damages
  the previous index.
- Model downloads skip weight formats the app never loads (SmolLM2-135M: 261 MB instead of 1.97 GB).
- Training presets keep their planned epochs on large datasets; Quick work picks the preset by model size; eval loss is
  labelled as no recall signal for fact datasets.

### Fixed

Smart Brain and packages
- A second new package of a project no longer overwrites the first one's file (the file name carries the package id).
- *New project* opens the new project; the Package page's *Build the package* lands on the export panel; an unbuilt
  Smart Brain page no longer logs a 404.
- The quiz's "unanswerable" questions are checked more carefully (answers by clear implication count), so a correct
  answer is less often marked fail.
- The standalone server no longer crashes when several clients search at once, runs exactly the search layers the
  index has on, bundles the chosen reranker, and works again after Re-embed only with another embedder and after
  opening a package made with models.
- An XML element, JSON object or YAML item that fits one passage is no longer cut in two.
- Importing an index works in a new project and says where it came from.

Testing and Compare
- A run whose every answer fails ends as failed; a stopped judge is shown as stopped, with the verdicts kept;
  stopping parallel API judging no longer queues the rest. Model folders no loader can open are refused up front.

Training, export and Heretic
- Phi-4-mini and Gemma 4 train; a NaN loss stops the run with the step instead of ending as done.
- *Merge on save* frees the training model first, so a 9B merges on the GPU in seconds.
- "Open in inference" loads the run's model (it always failed with a tokenizer error).
- Heretic: fixed batch size 8 (the auto probe crashed the GPU), a trial within the KL budget instead of trial 0 (which
  could produce token soup), progress "trial 13 of 40", and its exports appear on every model list.

Install, update and CLI
- The installer could build on any newer Python on PATH and lose the GPU; the health check no longer reports a missing
  driver or packages when it only ran out of time. Settings → Updates → Check really checks GitHub.
- `fts` run from another directory uses the app's own database; `fts models delete` and `fts update` refuse unsafe
  cases.

Interface
- The app no longer freezes in a tab after moving between pages with live updates (connections are closed on leave).
- Tab focus and jump links no longer hide controls under the top bar; destructive actions ask in the app's dialog;
  forms show that they are working and cannot be sent twice; many smaller label, count and empty-state fixes.
- OCR uses one CPU thread per file, so uploading many scans at once no longer grinds the machine.

### Removed

- The old helper and vision settings routes (`/api/settings/helper…`, `/api/settings/vision`; replaced by
  `/api/settings/assistant` and `/api/settings/connections`, saved choices carry over), the API-only quantize routes,
  API routes nothing called, the Windows installer scripts and the conda install mode.

### Security

- Saved API keys are stored encrypted (a copied database carries no usable key); older plain-text keys are encrypted
  on the next start.

## [0.2.0] - 2026-10-25

First tagged release (beta, Linux only). What the app does at this point, each part measured on the fictional Korvane
corpus (`tests/corpus/korvane/RESULTS.md`): Files & Knowledge (upload, parse, OCR, Vision reader) → per-file mining
into reviewed question/answer pairs → LoRA / QLoRA training (study set, replay) → Testing with an AI judge, paired
Compare (McNemar) and IFEval / MMLU benchmarks → Smart Brain retrieval and chat → GGUF export. Earlier builds were
numbered `0.1.0.N`, one per commit.

### Added
- Release system: `VERSION` holds the SemVer of the release line, `scripts/release.py` (`make release-prepare`,
  `make release-finish`) cuts `release/vX.Y.Z` from `dev`, and a pushed `vX.Y.Z` tag on `main` becomes a GitHub
  Release with its sdist, wheel and these notes.
- Branch model: `main` takes only releases, `dev` is the development branch, work happens on `<type>/<slug>`
  branches cut from `dev`. The required `branch-policy` check, GitHub rulesets and the `pre-commit` / `pre-push`
  hooks (`make hooks`) enforce it.
- `GET /api/system/version` also returns `release`, `prerelease` and `git_branch`.

### Changed
- The version in the UI is `X.Y.Z` on a tagged release and `X.Y.Z+<branch>.<sha>` on any other commit; the
  per-commit `0.1.0.N` build counter and its `pre-commit` bump are gone. The channel label reads BETA for 0.x.
- `update.sh` pulls the branch the checkout is on (`main` or `dev`) and leaves feature branches alone.
- Linux (NVIDIA GPU or CPU) is the only supported platform until 1.0.0: the install docs and the project page no
  longer offer Windows or macOS paths, and CI no longer runs a macOS leg. `docs/INSTALL.md` "Updating" now uses `update.sh`.

### Fixed
- Mining keeps the direction of comparisons as written (beat/missed, above/below, rose/fell …): a board pack's
  "beat the budget" had been mined as "missed". Mining prompts are now `mining-v5`.
- `uv build` failed because the benchmark fixtures were added to the wheel twice; the sdist no longer ships the
  25 MB test corpus.

### Removed
- The legacy `/api/data/{upload,validate,dedup,analyze,convert,hallucination-check}` routes: nothing in the UI or
  CLI called them (project files go through Files & Knowledge and Data prep; `fts analyze` / `fts validate-hallucination`
  are unchanged). Unused code: `templates.TemplateManager`, `HallucinationGuardrail` and a few helpers nothing called.

[Unreleased]: https://github.com/GenorTG/finetune-studio/compare/v0.3.0...dev
[0.3.0]: https://github.com/GenorTG/finetune-studio/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/GenorTG/finetune-studio/releases/tag/v0.2.0
