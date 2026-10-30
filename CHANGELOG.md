# Changelog

All notable changes to Finetune Studio. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and versions follow [Semantic Versioning](https://semver.org/) (see [CONTRIBUTING.md](CONTRIBUTING.md#versions-and-releases)).
0.x releases are betas; 1.0.0 will be the first stable Linux release.

Every feat/fix/perf pull request into `dev` adds its line under **Unreleased**; `scripts/release.py prepare`
turns that section into the release's notes.

## [Unreleased]

### Added
- Smart Brain answers can make a **follow-up search**. A question that reaches its answer through another item ("who manages the site where pump X is installed?") finds the pump's row, never the site's profile; the answer model may now ask for one more search before answering. The Smart Brain chat shows it as *Also searched*, and an exported package's `/ask` and `rag_answer` do the same (`RAG_FOLLOW_UP`, 0 turns it off). On the Quillmoor benchmark (376 judged answers, API model) two-step questions went from 10 % to 46 % with the old default search and to 84 % with the new reranker.
- **Heretic** (optional tool): remove a model's refusals properly. Heretic searches the abliteration parameters
  (Optuna) instead of applying one fixed strength, so the result stays closer to the original model's behaviour. It
  is a separate program with its own license (AGPL-3.0), installed on demand from Settings → **Heretic** (or
  `fts heretic install`): it reuses this app's PyTorch, so it costs a few hundred MB rather than another copy of
  CUDA. **Check upstream** compares it with Heretic's GitHub releases, and the Export page gains a **Heretic** format
  beside the existing Abliterated/Merged/GGUF ones. Until it is installed the option is greyed out and says why.
- CLI: `fts brain bundle` (encrypted `.ftsrag` copy of an index) and `fts brain import` (load one into a project).
- Smart Brain standalone export: **each search layer is its own switch**. You can turn off **Meaning search**
  (vectors), **Word search** (BM25), the **Reranker** and **Answer generation** separately. **Rank fusion** runs
  automatically when both searches are on.
  - The package only ships the files the layers you keep need: no vectors without meaning search, no keyword
    index without word search.
  - A **Pipeline** line shows how a question travels through the package.
  - The server reports in `/health` which layers really run. It can switch a shipped layer off at run time
    (`--no-semantic`, `--no-keyword`), but never on one the package does not contain.
  - **Answer generation** (the "G" in RAG) adds `/ask` and the MCP tool `rag_answer`. They send the passages to any
    OpenAI-compatible chat model with the same grounded prompt as the Smart Brain chat. The API key is never put
    in the package.
  - New CLI: `fts brain export` with the same switches.
- Buttons for four things that only worked through the API:
  - **My models → Rescan folders** finds a model you copied into a model folder by hand, without restarting the app.
  - **Smart Brain → Re-embed only** swaps the embedder of an existing index and keeps its passages. It is quicker
    than rebuilding from scratch.
  - **Q&A pairs → Remove near-duplicates** rejects pending pairs that repeat a fact another pair already asks. It
    shows the count and asks before rejecting anything.
  - **Q&A pairs → Data audit** re-reads every file from its stored bytes. It reports files whose parse or passages
    changed, and approved pairs that point at a missing passage. No model is involved.
- Training page: **Start from a use case**. Pick extraction, classification, house style or distillation, upload
  examples, and the app checks every row (unusable ones are listed by row number), keeps a held-out test, fills the
  training form with settings sized to the data, and after training compares the run with its untrained base on that
  test (paired McNemar, or the blind A/B judge for style). `?usecase=<id>` opens the page with the form filled.
- Quick work, Step 6: saved versions are listed with what each one holds (datasets, training runs, Smart Brain
  build). A new version **builds on** an earlier one (the latest by default), **Lineage** shows the chain
  (v1 → v2 → v3), and **Delete** removes a version record without touching its datasets, runs or models.

- Export page: **Importance matrix (imatrix)** for GGUF. Before quantizing, the app measures which weights the
  run's own training data relies on (llama.cpp's `llama-imatrix`, sized to the free GPU memory) and quantizes with
  that: better q4/q3/q2 quality at the same file size. The files go to the run's `gguf-imatrix/` folder beside the
  plain ones; the option is disabled, with the reason, when the tool is missing.

- Settings: **AI assistant** replaces the separate Helper model and Vision reader cards. One main assistant does every
  helper job (Guide chat, mining, study sets, reworded quizzes, preference pairs, test judge, use-case teacher,
  reading pictures). Add any number of API connections next to the local model, each with its own key (stored
  encrypted, sent only to its own URL, never shown again); pick the main assistant; give any job its own connection
  and extra instructions; the jobs table says what each job sends to an API. API connections no longer take the
  one model slot, so asking the Guide something on an API assistant leaves the local model you loaded alone.

- Training page, use cases: **draft examples with the AI assistant**. Upload raw texts (tickets, invoices, emails that
  already sound right); the assistant drafts the label, the extracted JSON or the request each text answers; you fix,
  approve or reject every draft, and only approved rows are built into the use case. Settings → AI assistant has a
  new job for it (*Drafting use-case examples*), with its own connection and instructions like every other job.

- The **Guide** (corner panel on every page) can now do the work, not only explain it: start mining, build the
  dataset, build the Smart Brain, start training, build a reworded quiz, run a test and pin a version. Choose in the
  Guide's header: **Ask before each action** (default: an action card with Run / Skip) or **Hands-free** (it runs the
  actions itself and carries on with the next step, up to 4 in a row per message). Actions go through the same
  endpoints as the page buttons; the Guide still never approves pairs, exports a model or deletes anything.

- Activity drawer: a running model download has a **Cancel download** button (the API to stop one existed with no
  way to reach it from the page), and a cancelled job shows *cancelled* instead of *error*.

### Changed
- New Smart Brain indexes on a machine with a GPU rerank with **bge-reranker-v2-m3** (run in half precision) instead of ms-marco-MiniLM: on the Quillmoor benchmark it found the right passage in the top 5 for 89 % of questions instead of 79 % (reverse look-ups such as "which asset has serial SN…": 40/40 instead of 24/40) at about 0.2 s per search, and the API model's judged answers rose from 81.1 % to 88.4 % (96.1 % with follow-up search). Machines without a GPU keep MiniLM, which stays fast on a CPU; existing indexes keep the reranker they have (Search settings switches it, no rebuild). The reranker is a ~2.2 GB download, and a standalone export with models grows to about 4.5 GB.
- The dataset check no longer calls one-word answers a problem for a use-case dataset (a label is short by design).

### Fixed
- Guide and page copy: informational answers survive readiness checks, current-page help wins relevant searches, unrequested page switches are blocked, Compare explains its installed-model fallback, and Pairs/Testing/Benchmarks/Export labels match the flow (steps 1/3/3/4).
- Guide: each request now includes the complete current-route KB entry ahead of stale chat history; Dashboard, Project Settings and Settings facts are corrected, and Training now identifies step 2 and says training loss is not held-out quality.
- Smart Brain standalone server: several clients searching at once no longer crash it. Parallel requests ran the
  bundled embedder and reranker on the GPU at the same time ("CUDA error: an illegal memory access" after a few
  hundred searches) and could load a model twice; model loading and use are now one at a time.
- Training stops and fails in plain words when the loss or the gradient becomes NaN ("Training diverged at step N
  ... the last checkpoint before it is ...") instead of finishing as **done**. Such a run used to log a loss of 0.0
  for its remaining steps and merge and export weights that answer nothing.
- Smart Brain: after **Re-embed only** with another embedder, exporting the index **with models** works again (it
  refused: "does not reference a shared embedder"). A reranker chosen in **Search settings** is now bundled into an
  exported package (it was left out without a word, so the package ran without reranking), and rebuilding the
  index keeps the chosen reranker instead of resetting it to the default one.
- Uploading many pictures or scanned PDFs at once no longer grinds the machine: every OCR run now uses one CPU
  thread. With a dozen files parsing in parallel, one small photo took 137 s to read and could hit the 180 s
  limit; it now takes about 2 s.
- Heretic export keeps a trial that actually works: the KL budget default drops from 0.5 to 0.1. With 0.5 the
  end-to-end run kept a trial at KL 0.287 and the exported model repeated "(Please, choose a few questions…" forever,
  while another trial of the same study at KL 0.0028 answered normally. KL is not comparable across models, so the
  budget has to suit the worst case; 0.1 still admits trials around KL 0.07 at 7% refusals on a 1.5B model.
- Heretic export can pin the batch size (`--batch-size`). Heretic's auto probe picked 128 sequences per pass for a
  1.5B model, and the run died mid-search with `CUDA error: an illegal memory access` inside the L-BFGS step. It is
  reported as a failed export rather than a fake success, but an export that cannot finish is still a gap; with
  `batch_size=8` the same 24-trial run completed in 48 min at a steady ~8 GB.

- Heretic export picks the trial to keep by a KL-divergence budget instead of trial index 0. Index 0 of Heretic's
  Pareto front is the fewest-refusals trial **at any cost**: on a real run it had KL 10.0 and the exported model
  answered in token soup, so the refusals were gone because the model was. The export now searches, reads the front,
  keeps the trial with the fewest refusals inside a KL budget (now 0.1), and says so when no trial meets it.
- Smart Brain **Import an index from another studio** is its own card, open in every project. Before, it was
  inside the export panel, which only appears once the project has an index, so a new project could not import a
  bundle. An imported index now says where it came from instead of "rebuild needed: N file(s) removed", and it is
  registered like a build, so the Guide and the Testing page see it.
- Settings → Updates: **Check** now asks GitHub whether a newer version exists and lists what changed. Before, it
  never looked, although the button promised to. **Apply update** reports which version it moved from and to.
- The install health check (Check, `update.sh`, `install.sh --check`) no longer gives two false alarms:
  - "no working NVIDIA driver: install it and reboot" when the driver was loaded and only `nvidia-smi` was
    missing from the app's PATH. It now reads the loaded driver.
  - "missing or broken: peft" (or torch, or the core packages) when an import just took longer than the time
    limit on a busy machine. That is now "could not check", never "missing".
- Smart Brain builds can be stopped: a **Stop build** button on the page, and `fts jobs stop` now works too. Before,
  the whole index was embedded in one step that ignored stop requests. A stopped or failed build, including a
  rebuild from scratch, now leaves the previous index untouched; a rebuild used to delete it first.
- The Smart Brain build bar shows the passages embedded, the real time taken and the time left. Before, it showed
  "0 files, 0 s" until a whole file was done.
- A build that runs out of GPU memory now says so in plain words: how much memory was free, which programs hold the
  card, and the choices (a smaller embedder, the CPU, or waiting). Before, it showed the raw CUDA error.
- The standalone Smart Brain server ignored the index's "hybrid off" setting and always ran keyword search; it
  now runs exactly the layers that are on. Its "Reranker on" option did nothing without bundled models; the export
  page now says so and only offers it with *Include models*.
- Returning to the Projects page from another page after a full page load no longer stops its script with a duplicate-declaration error.
- Cancelling a whole-model download stopped the progress bar but not the download: the job process stayed alive and
  fetched the rest of the repository (measured: 22 GB cancelled, still writing ~70 MB/s). The job now exits when it
  is stopped, and the Activity drawer's **Cancel download** button sits on the row itself and still works while the
  row refreshes every second.
- The pages, the Guide's knowledge base and error messages now call the model that mines, judges and drafts the
  **AI assistant**, as Settings does, instead of the old "helper model" / "helper seat"; `/openapi.json` reports the
  app's real version instead of 0.1.0.
- A job that needs the local assistant (drafting use-case examples, the reworded quiz, study sets, preference pairs)
  no longer stops at "no helper model loaded" with no way forward: the message now has a **Load the assistant and
  retry** button that loads the job's local model and runs the job again.
- Quick Work keeps the selected preset on tiny datasets and labels sub-100-step runs as workflow smoke tests, instead of silently stretching them into severe over-training. It identifies its generated quiz as training-set recall (not held-out generalization), recovers missing merged weights through the supervised Export job, and only announces completion when testing succeeds.
- Quick work, Step 6: after pinning a version, **Builds on** moves to that new version, so the next pin continues
  the line instead of branching from the one before.
- Testing / Compare: picking a folder no loader can open (a training run's own folder, with only `adapter/` and
  checkpoints) is refused at once with what to do (export the run first), instead of starting a run that failed
  inside the job with a tokenizer error.

### Removed
- `GET/PUT /api/settings/helper`, `POST /api/settings/helper/models` and `GET/PUT /api/settings/vision`: replaced by
  `/api/settings/assistant` (main assistant, jobs) and `/api/settings/connections` (API connections). Saved helper,
  vision and judge choices carry over (the old `api-helper` / `api-vision` rows become connections).
- `POST /api/training/runs/{id}/quantize`, `GET /api/training/runs/{id}/quant-exports` and
  `POST /api/training/runs/{id}/set-output`: API-only, never used, and the quantize route needed an importance-matrix
  file nobody had. The Export page's importance-matrix option replaces it. The hidden `export_imatrix` /
  `imatrix_calibration` training fields are gone too.
- The unmaintained Windows installer and launcher scripts (`install.bat`, `install.ps1`, `run.bat`): Linux is the
  only supported platform. Windows support will come back as a cleanly tested installer after 1.0.
- The conda install mode (`USE_CONDA=1`, `make install-conda`): it installed torch and llama.cpp into `.venv`,
  which conda mode never created, so it could not produce a working install. Use `bash install.sh` (`.venv`).
- API routes nothing in the app called: `POST /api/testing/load` (use `/api/inference/load`),
  `GET /api/benchmarks/suites`, `GET /api/benchmarks/projects/{pid}/runs/{rid}/history`,
  `POST /api/projects/{pid}/data-prep/chat` and `.../data-prep/chat/tools` (the Guide replaced them),
  `GET /api/guide/kb`, `/api/guide/kb/{id}` and `/api/guide/help` (the Guide reads its help in-process).

### Security
- Saved API keys (helper, Vision reader) are now stored encrypted instead of as plain text: a copied database or
  backup carries no usable key. Keys saved by earlier versions are encrypted on the next start; Settings shows
  "saved, encrypted", or asks for the key again if the encryption key file was lost. A plain-text judge key left in
  `settings.json` by an old version is deleted.

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

[Unreleased]: https://github.com/GenorTG/finetune-studio/compare/v0.2.0...dev
[0.2.0]: https://github.com/GenorTG/finetune-studio/releases/tag/v0.2.0
