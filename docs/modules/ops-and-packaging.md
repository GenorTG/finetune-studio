# Operations and packaging — developer reference

This map distinguishes canonical repository workflows from legacy helpers.
Read the script before running it: installers and updaters can change the
Python environment or restart the app service.

## Canonical paths

| Entry | Role |
|---|---|
| `install.sh` | Canonical Unix install/repair path; establishes `.venv`, torch constraints, parser dependencies, and project-local llama.cpp. |
| `update.sh` | Canonical in-place update path; reuses the checkout, repairs/syncs dependencies, checks llama.cpp, migrates the DB, then restarts the user service. |
| `install.bat`, `install.ps1` | Windows entrypoints; require Python 3.12–3.13 and install the `parsers` extra. |
| `install-service.sh` | Generates the user systemd unit using the current checkout's venv and configured host/port. It does not delegate to `scripts/run.sh`. |
| `scripts/install_diagnose.py` | Read-only diagnostics by default; `--repair` enables repair actions. `--json` emits a JSON array (and repair action object when combined with `--repair`). |
| `scripts/codemap.py` | Generates `docs/CODEMAP.md`; `make codemap` is the repo wrapper. |

## Legacy and support utilities

- `scripts/run.sh` is a legacy user-settings launcher. It now resolves the venv
  relative to this checkout; the generated service does not use it.
- `scripts/update.sh` is a separate tag/ref updater with `--check`, `--force`,
  and target-ref behavior. It is not the canonical `update.sh`; no in-repo caller
  was found during the audit. Its default checkout path remains a legacy concern;
  use root `update.sh` for normal deployment.
- `scripts/finetune-studio.service` is a checked-in host-specific historical
  unit. Prefer the generated unit from root `install-service.sh`.
- `scripts/install-hooks.sh` installs repository hooks for local commits.
- Dataset/corpus builders and probes (`augment_dataset.py`,
  `build_full_qa_suite.py`, `build_source_disjoint_split.py`,
  `generate_dataset.py`, `gen_vaelindrath_corpus.py`, `gguf_layer_probe.py`,
  `capture_shots.py`) are development utilities, not install/deploy entrypoints.

## Shell safety and verification

- The Windows installers and fallback `requirements.txt` must stay aligned with
  `pyproject.toml` and the `parsers` dependency group.
- `scripts/run.sh` expects an already-created checkout-local `.venv`; it now
  fails with an install hint instead of invoking a hard-coded `$HOME` path.
- `tests/run_qa.sh` is a live remote UI run and can load models or mutate project
  data. It exits before network contact unless `FTS_ALLOW_LIVE_E2E=1` is set.
- Validate shell syntax with `bash -n` on changed shell scripts. Do not run the
  live E2E runners as part of ordinary unit-test verification.
