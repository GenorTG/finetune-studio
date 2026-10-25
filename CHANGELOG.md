# Changelog

All notable changes to Finetune Studio. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and versions follow [Semantic Versioning](https://semver.org/) (see [CONTRIBUTING.md](CONTRIBUTING.md#versions-and-releases)).
0.x releases are betas; 1.0.0 will be the first stable Linux release.

Every feat/fix/perf pull request into `dev` adds its line under **Unreleased**; `scripts/release.py prepare`
turns that section into the release's notes.

## [Unreleased]

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
- `uv build` failed because the benchmark fixtures were added to the wheel twice; the sdist no longer ships the
  25 MB test corpus.
