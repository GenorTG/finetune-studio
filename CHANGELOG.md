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

### Changed
- The dataset check no longer calls one-word answers a problem for a use-case dataset (a label is short by design).

### Security
- Saved API keys (helper, Vision reader) are now stored encrypted instead of as plain text: a copied database or
  backup carries no usable key. Keys saved by earlier versions are encrypted on the next start; Settings shows
  "saved, encrypted", or asks for the key again if the encryption key file was lost.

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
