# Changelog

All notable changes to Finetune Studio. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and versions follow [Semantic Versioning](https://semver.org/) (see [CONTRIBUTING.md](CONTRIBUTING.md#versions-and-releases)).
0.x releases are betas; 1.0.0 will be the first stable Linux release.

Every feat/fix/perf pull request into `dev` adds its line under **Unreleased**; `scripts/release.py prepare`
turns that section into the release's notes.

## [Unreleased]

- Image search answers better: the answer model is told the text around each matching picture (its own heading and
  caption, slide or cell text), Excel pictures carry the cells around them, plural words match ("plates" finds a plate), and a build
  re-extracts pictures when the extraction itself improved. Long file names wrap in the picture results.
- Packages can ship **Image search** (Package → search layers, with Include models): the pictures of the package's
  documents as vectors and thumbnails inside the encrypted corpus plus the CLIP model; `POST /search/image` (text or
  `image_base64`), `GET /images/<id>/thumb`, the MCP tool `rag_search_images` (thumbnails as image content), an
  *Attach a picture* button on the chat page, and `/ask` passes matching or attached pictures to vision-capable answer
  models (OpenAI and Claude image parts) with a text-only fallback. A scoped package carries only its own pictures;
  the self-test finds a known picture by words and by the picture; `/version` lists the `image` layer.
- **Image search** (Smart Brain → Search settings): pictures in your files — photos, scans, charts and the images
  inside Word, PowerPoint, PDF and Excel files — are embedded with a pinned open-licence image/text model (CLIP
  ViT-B/32 with a multilingual text encoder by default, MIT + Apache-2.0, runs on a CPU), each with a thumbnail, its
  file and page/slide/paragraph/sheet, and the OCR or Vision-reader reading of the same picture. Find a picture by
  describing it (50+ languages) or by uploading one (also lists the documents it is in); Test a search and the chat
  show matching pictures with their own scores; open, download or go to the source file. Builds embed only new or
  changed pictures; the Knowledge check lists pictures that could not be embedded or that nothing describes.

- Packages have unique, dated names: the archive and its folder are `<project>-<package>-v<version>-<date>`, and the
  service (`setup.sh --service`) and MCP server are named after the package (`<package>-<id8>`, the same for every
  version), so two packages on one host no longer collide as `rag-pack`. A copy installed before keeps working on
  update; `bash update.sh --rename` moves it to the new name. The running package reports its name, version and
  build date on a public `GET /version`, in `/health`, an `X-Package-Version` header, MCP `serverInfo`, `rag_info`,
  the chat page and the README; the package check compares them with `package.json`.

## [0.5.0] - 2026-11-13


- The Guide answers "Is my knowledge ready to package?" from the Package page's own checklist (files read,
  knowledge check, Smart Brain up to date, then identity, test and package); the training pipeline follows as
  optional instead of being the answer.
- Files: the upload result names each file that was not added and why, and the duplicates it skipped; the selection
  tools appear once a file is ticked; the response holds no server paths. The knowledge check lists files no parser
  can read (`.exe`, `.bin`, …) and flags a password even next to an API key in the same file.
- The Package page says, next to the Build button, how the package will search where it runs: a package without
  the models inside searches by exact words only until the host sets an embeddings endpoint, and the form now takes
  that endpoint (`--embed-base-url` in the CLI) so meaning search works there. The self-test result says the same
  instead of only reporting the search it ran with the studio's own embedder.
- A package's `rag.config.json` turns the reranker on only when its model is inside, and its README no longer
  says the original files are inside when they are not.
- A package exported with meaning search off no longer gets the self-test note about needing an embedder.
- Testing: a Smart Brain test can use an AI assistant API connection as the reader (nothing loads on the GPU), so
  the package's answers can be tested without a GPU; a project without exports starts on that connection instead of
  a "latest merged export" it does not have.
- The Testing, Training, Quick work and Smart Brain chat model lists show this project's exports and installed
  models only, each model once, never another project's trained runs.
- A project can no longer be renamed to a blank (or over-long, or another project's) name: Project settings and the
  API refuse it with the reason. New: `fts projects rename <project> "<name>"`.
- CLI: an unknown project or another user error is one line ("project not found: … (fts projects list shows them)"),
  never a Python traceback; `fts brain export --out NAME.tar.gz` writes that file instead of a folder of that name,
  and the CLI's help calls itself `fts` and leads with the knowledge package.

- Package versions are safe to roll out: a new version keeps its package's documents, original files, hidden file
  names and audience unless you change them, anything wider (more documents, originals on, names shown, a public
  audience changed, encryption off) needs a confirmation (`--allow-widen` in the CLI) and is listed in CHANGES.md,
  and `update.sh` refuses such a version on a running copy unless `--allow-widen` is given.
- Reading pictures with a local Vision reader no longer happens silently: the upload's note names the model and
  about how much GPU memory it loads, the top bar shows it loading, it does not start without enough free GPU memory
  or over a model loaded for Chat or a test, and the Smart Brain says when files were re-read by the Vision reader
  after the build instead of just "N changed".
- A new version of an encrypted package needs the current passphrase (or a new one on purpose), so running copies
  still start after `update.sh`; `update.sh` refuses an encrypted version it cannot open first, and uses
  `passphrase.txt` beside `server.py` (which `setup.sh` now offers to write) when no `--keyfile` is given.
- The package's chat page answers without an answer model too (the matching passages; file names stay hidden for a
  public bot), Enter asks, its sources are the files the answer names; a busy port is one line with the fix; in the
  studio's Smart Brain chat a declined answer lists what was searched instead of "Sources"; connection tests say
  "connection refused" / "host not found" in words and show the failure next to the form.
- `update.sh` takes the new version's settings (answer model, hidden names, …) and keeps only the `rag.config.json`
  values you changed after install, and prints both; relative archive and keyfile paths work from any folder.
- The Smart Brain quiz checks an "unanswerable" question against the passages sharing its words as well as the top
  search hits, so an answer split over two files is found; on a test run, **The documents answer this** fixes one it
  still missed (your Pass, the model's answer as key, the quiz file corrected for the next run).
- After a package build the Package page goes back to "A new package" instead of silently making the next export a
  new version of the one just built.
- Smaller fixes: `/projects/<id>/package` opens the Package page; the Build button is off (with the reason) while
  superseded documents block packaging; the Smart Brain no longer says the first build downloads 4 GB when the models
  are on disk; status pills stay pills on phones; every form control has a name for screen readers; a "Skip to
  content" link comes first for keyboard users.
- The package README's Claude Code, Codex and Cursor MCP commands include `--keyfile` for an encrypted package.
- Opening a package keeps its name (from the package, or the file name you chose) instead of a temporary
  `upload-<number>.tar.gz` name in Package versions, the base list and the next file name.
- Importing a damaged or wrong file is refused with a plain reason (HTTP 400, not 500), and the file and its
  passphrase are checked before you are asked to replace the current index.

## [0.4.0] - 2026-11-12

- New projects' optional base-model list now excludes inference-only and other projects' exported models, and de-duplicates identical paths.
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

[Unreleased]: https://github.com/GenorTG/finetune-studio/compare/v0.5.0...dev
[0.5.0]: https://github.com/GenorTG/finetune-studio/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/GenorTG/finetune-studio/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/GenorTG/finetune-studio/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/GenorTG/finetune-studio/releases/tag/v0.2.0
