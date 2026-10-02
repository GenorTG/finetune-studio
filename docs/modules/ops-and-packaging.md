# Operations and packaging — developer reference

This map follows the checked-in entrypoints and their actual callees. Install,
repair, update, and service commands can alter the environment or restart the
app; inspect their flags before running them.

## Canonical lifecycle

| Entry | Verified behavior |
|---|---|
| `install.sh` | Detects OS/GPU/Python, creates or repairs `.venv`, installs the matching Torch family and constraints, editable app + parser dependencies, llama-cpp-python and project-local llama.cpp CLI, then runs diagnostics. It does not install a systemd service or browser-test dependencies. |
| `run.sh` | If needed, calls `install.sh`, activates the checkout venv (or configured Conda env), then `exec`s Uvicorn for `finetune_studio.webui.app:app`; defaults to host `0.0.0.0`, port `7860`. |
| `update.sh` | Pulls `origin/main` with `--ff-only`, diagnoses/repairs dependencies, syncs the editable package while preserving Torch constraints, checks/builds llama.cpp, initializes the DB, and restarts an active user service. `--check` still performs the configured pull and DB initialization; it is not a read-only dry run. Repair may recreate the venv. |
| `install-service.sh` | Separate Linux action. Generates the systemd user unit from this checkout's venv and host/port settings; supports status/restart/uninstall. Not called by normal `install.sh`. |

The supported CLI launcher is `fts webui --host HOST --port PORT` (or
`python -m finetune_studio webui`). `finetune_studio.webui.app` defines the
ASGI object but has no module entrypoint that starts Uvicorn. On Windows,
`run.bat` launches the same CLI subcommand after activating `.venv`; fish and
zsh launchers delegate to root `run.sh`.

## Script inventory

| File | Role / caller |
|---|---|
| `scripts/install_diagnose.py` | Read-only-by-default install health inspection; `--repair` runs repair actions. Called by install/update paths. Some repair branches still invoke `venv-python -m pip`; this can fail for a uv-created venv without the `pip` module. Install/update use an uv-first installer helper. |
| `scripts/install-service.sh` | Thin wrapper to root `install-service.sh`. |
| `scripts/run.sh` | Legacy user-settings launcher; reads `~/.finetune-studio/settings.json` and launches Uvicorn from the checkout venv. The generated unit does not use it. |
| `scripts/codemap.py` | Stdlib AST index generator for `docs/CODEMAP.md`; default invocation writes, `--stdout` prints, `--grep NAME` searches the generated map. |
| `scripts/git-hooks/pre-commit` + `scripts/install-hooks.sh` | Installed by `make hooks`; pre-commit increments the BUILD component in `VERSION` and stages it, unless `FTS_NO_BUMP=1` or the commit only changes `VERSION`. |
| `scripts/augment_dataset.py` | Source-grounded augmentation for a project's approved Q&A pairs; writes augmented pairs, held-out suite, and refreshed ShareGPT JSONL. |
| `scripts/build_full_qa_suite.py` | Thin wrapper around `testing.full_corpus_suite` to write a suite from every eligible project pair. |
| `scripts/build_source_disjoint_split.py` | Groups approved pairs by source ID and writes a source-disjoint training split plus held-out benchmark suite. |
| `scripts/capture_shots.py` | Playwright capture utility for a configured live host/project; writes page screenshots under `docs/screenshots/`. |
| `scripts/gen_vaelindrath_corpus.py` | Generates a synthetic multi-format fictional corpus and its ground-truth facts for parser/training evaluation. It has module-level output-directory setup; inspect the `OUT` constant before invoking. |
| `scripts/generate_dataset.py` | Generates generic synthetic chat training JSONL from prompt templates. |
| `scripts/gguf_layer_probe.py` | Prints GGUF metadata fields related to layer counts/context length. |

Additional OS entrypoints (`install.{bat,fish,ps1,zsh}` and `run.{bat,fish,zsh}`)
either implement the platform-specific install or delegate to the canonical
Unix shell scripts. Keep launch examples aligned with the registered `webui`
CLI command. `requirements.txt` is a fallback dependency list; the authoritative
package metadata is `pyproject.toml`.

## Verification boundaries

- `Makefile`'s `lint` target historically masked Ruff failure with `|| true`;
  use `.venv/bin/ruff check src/ scripts/` and inspect its exit code. A broader
  current audit also included tests and docs separately; see the dated audit
  ledger rather than treating this map as a lint report.
- `tests/run_qa.sh` is an opt-in live UI runner; it has remote-service and data
  side effects. `tests/README_E2E.md` documents its guard and setup. Do not
  run it as if it were a local unit test.
- `scripts/install_diagnose.py --repair`, root `update.sh`, and
  `install-service.sh --restart` can mutate the venv/database or service.
- There is no tracked `scripts/update.sh` or `scripts/finetune-studio.service`;
  the canonical updater and generated service installer are at the repository
  root.
