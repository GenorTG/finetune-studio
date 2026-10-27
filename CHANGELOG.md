# Changelog

All notable changes to Finetune Studio. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and versions follow [Semantic Versioning](https://semver.org/) (see [CONTRIBUTING.md](CONTRIBUTING.md#versions-and-releases)).
0.x releases are betas; 1.0.0 will be the first stable Linux release.

Every feat/fix/perf pull request into `dev` adds its line under **Unreleased**; `scripts/release.py prepare`
turns that section into the release's notes.

## [Unreleased]

### Added
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
- The dataset check no longer calls one-word answers a problem for a use-case dataset (a label is short by design).

### Fixed
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
