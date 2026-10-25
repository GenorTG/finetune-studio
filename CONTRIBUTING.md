# Contributing

Finetune Studio is built for **Linux** (NVIDIA GPU or CPU). The goal is a stable Linux 1.0.0; other platforms
are out of scope until then.

## Branches

```
feat/… fix/… chore/… ──PR──▶ dev ──release/vX.Y.Z──PR──▶ main ──tag vX.Y.Z──▶ GitHub Release
                              ▲                            │
                              └──────── sync/vX.Y.Z ◀──────┘
```

| Branch | What it is | Changes only by |
|---|---|---|
| `main` | Released code. Every commit on it is a release or a hotfix; users clone and `update.sh` this. | a merged PR from `release/vX.Y.Z` or `hotfix/vX.Y.Z` (merge commit) |
| `dev` | The development branch: everything that will be in the next release. | a merged PR from a work branch (squash) or `sync/vX.Y.Z` (merge commit) |
| `<type>/<slug>` | One piece of work, cut from `dev`. `type`: `feat` `fix` `chore` `docs` `refactor` `test` `ci` `perf` `build`; `slug`: lowercase `a-z 0-9 . _ -`. | you; deleted after merge |
| `release/vX.Y.Z` | A release being cut from `dev` (VERSION bump + dated CHANGELOG). | `scripts/release.py prepare` |
| `hotfix/vX.Y.Z` | An urgent fix cut from `main`. | you, then `scripts/release.py prepare patch` |
| `sync/vX.Y.Z` | Brings a release back from `main` into `dev`. | `scripts/release.py finish` |

Hard rules, enforced in three places:

1. **No direct commits or pushes to `main` or `dev`.** Branch from `dev`, open a PR. GitHub rulesets refuse direct
   pushes, force pushes and deletion of both branches; the `pre-commit` and `pre-push` hooks refuse them locally.
2. **`main` takes only `release/vX.Y.Z` and `hotfix/vX.Y.Z`.** The branch's VERSION must equal X.Y.Z, be newer than
   main's, and CHANGELOG.md must have its dated section. Merge with a **merge commit** (never squash), so `dev`
   and `main` keep one history.
3. **`dev` takes `<type>/<slug>`, `sync/vX.Y.Z` and `dependabot/*`.** A work PR never touches `VERSION`;
   `feat/`, `fix/` and `perf/` PRs add a line under `## [Unreleased]` in CHANGELOG.md (or carry the
   `skip-changelog` label).
4. **Required checks:** `ci-ok` (lint, codemap, install plan, the full test suite) and `branch-policy`
   (`scripts/branch_policy.py`) must pass before any merge.
5. **Never push the local history archive** (`full-history`, `backup/*`, `archive/*`). The repository is public.
6. Tags are `vX.Y.Z` only. Once pushed they are immutable (a ruleset refuses moving or deleting them).

Set up a clone once: `make hooks`.

## Everyday work

```bash
git switch dev && git pull --ff-only
git switch -c fix/mining-direction          # one logical change per branch
# … edit, then the module's tests + lint:
.venv/bin/python -m pytest -q -p no:cacheprovider tests/test_<module>.py
.venv/bin/ruff check src/ scripts/
# add a line under "## [Unreleased]" in CHANGELOG.md (feat/fix/perf)
git commit -m "Mining: keep beat/missed as written"   # imperative: what + why
git push -u origin fix/mining-direction
gh pr create --base dev --fill
gh pr merge --squash --delete-branch        # after ci-ok and branch-policy are green
```

Commit messages: imperative, saying what changed and why, one logical change each.
CHANGELOG lines are for users: what they can now do, or what no longer breaks.

## Versions and releases

[Semantic Versioning](https://semver.org/): `MAJOR.MINOR.PATCH`, optionally `-rc.N`.

- **MAJOR**: a change that breaks existing data, projects, exported Smart Brain packages, the CLI or the HTTP API.
- **MINOR**: new features, backwards compatible.
- **PATCH**: fixes only.
- `0.x` releases are betas (GitHub pre-releases). **1.0.0 = the first stable Linux release.**

`VERSION` holds the SemVer of the release line; `pyproject.toml` carries the same number. Only
`scripts/release.py` changes them. The app shows `X.Y.Z` on a tagged release and `X.Y.Z+<branch>.<sha>` on any
other commit (`GET /api/system/version`), so a dev checkout never looks like a release.

Cutting a release:

```bash
make release-status                 # what dev holds since the last tag, and the Unreleased notes
make release-prepare BUMP=minor     # release/vX.Y.Z from origin/dev: VERSION, pyproject, dated CHANGELOG, PR -> main
# wait for ci-ok + branch-policy, then merge with a MERGE COMMIT:
gh pr merge release/vX.Y.Z --merge --delete-branch
make release-finish V=X.Y.Z         # tags vX.Y.Z on main; the release workflow publishes the GitHub Release;
                                    # opens sync/vX.Y.Z -> dev
gh pr merge sync/vX.Y.Z --merge --delete-branch
```

Hotfix (a release is broken and dev is not ready):

```bash
git fetch && git switch -c hotfix/vX.Y.(Z+1) origin/main
# fix + test + CHANGELOG line under Unreleased, commit
python3 scripts/release.py prepare patch   # bumps on the hotfix branch, PR -> main
# merge (merge commit), then: make release-finish V=X.Y.(Z+1)   (syncs the fix into dev)
```

The release workflow (`.github/workflows/release.yml`) checks that the tag is on `main` and agrees with
VERSION, pyproject and CHANGELOG, builds the sdist and wheel, and publishes the GitHub Release with that
CHANGELOG section as its notes.
