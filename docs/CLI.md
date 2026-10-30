# Command-line reference

The `fts` command operates on the same local project database as the WebUI.
Start with `fts --help` to see every installed command and `fts <command> --help`
for that command's options. Commands that run locally do not require the WebUI
server unless their help says otherwise.

## Models

`fts models` without an action keeps listing installed models. Use its library
actions to search Hugging Face, manage background downloads, and remove only
models in Finetune Studio's own cache:

```bash
fts models
fts models search "Qwen3 0.6B" --limit 10
fts models search "Qwen3" --task "" --json
fts models local --json
fts models download hf-internal-testing/tiny-random-gpt2
fts models download-status <job-id>
fts models cancel-download <job-id>
fts models delete hf-internal-testing/tiny-random-gpt2  # asks for confirmation
fts models delete hf-internal-testing/tiny-random-gpt2 --yes --json
```

- `search` accepts optional query text, `--task`, `--library`, `--sort` and
  `--limit` (1–100). Hugging Face must be reachable.
- `download` queues a persistent background job and prints its id. `--file`
  downloads one repository file; otherwise the full repository is downloaded.
  Check progress or cancel it with the job id. The WebUI Activity drawer shows
  the same job.
- `local` lists only the app-managed Hugging Face cache. `delete` removes only
  that cache entry, asks before removing files, refuses non-interactive use
  without `--yes`, and protects a model currently loaded, loading, or still
  downloading. Trained exports, helpers and models elsewhere on disk are not
  affected. In JSON output, `hub_repo_id` is the original repository id when
  the cache folder can be identified unambiguously.
- All actions accept `--json`; errors exit nonzero with a plain-language reason.

The WebUI Model Library shows the same downloaded cache and provides a
confirmation before deleting a local download.

## Projects

List, inspect, create, and delete projects from a terminal:

```bash
fts projects list
fts projects list --json
fts projects show "Project name"
fts projects show <project-id> --json
fts projects create --name "Maintenance notes" \
  --description "Equipment manuals and maintenance procedures" \
  --base-model /path/to/base-model
fts projects delete "Maintenance notes"  # asks for confirmation
fts projects delete <project-id> --yes --json
```

- `list` shows each project's name, id, and base model. With `--json`, it
  also includes the last-updated timestamp.
- `show` accepts a project id or name and prints its description, base model,
  and system prompt. Names match case-insensitively; if more than one project
  matches, use its id.
- `create` requires a non-empty `--name`; description, base model, and system
  prompt are optional. Existing names are rejected case-insensitively unless
  `--allow-duplicate` is provided.
- `delete` accepts an id or name and removes the project record, file library,
  datasets, Brain index, run outputs, and exported models. It asks for explicit
  confirmation on a terminal; scripts must pass `--yes`. `--json` returns the
  deleted project's id and name. A missing project exits nonzero without changing data.
- `list`, `show`, `create`, and `delete` accept `--json` for machine-readable output.

The same records and deletion behavior are available on the WebUI's **Projects**
page. Deletion is permanent and removes that project's stored data from the
configured local Finetune Studio data directory and output directory.

## Files

Upload, inspect, and parse files in a project's shared file library:

```bash
fts files upload --project "Maintenance notes" ./manual.pdf ./faq.md
fts files upload --project <project-id> ./manual.pdf --json
fts files list --project "Maintenance notes"
fts files list --project <project-id> --mime text/ --json
fts files show --project <project-id> "maintenance.txt" --json
fts files parse --project <project-id> <file-id>
fts files parse --project <project-id> <file-id> --reparse
```

- `list` shows file metadata and parse status. `--search`, `--mime`, and
  `--include-deleted` filter the result; `--project` accepts an id or name.
- `show` accepts an id or exact file name. If multiple files share a name, use
  the id. Files in trash require `--include-deleted`.
- `parse` stores readable text and passages in the same library used by the
  WebUI. It reports parser errors and exits nonzero. `--reparse` is explicit
  because it discards a hand-edited parsed version and regenerates it from the
  original file.
- `upload` accepts one or more local paths. Readable formats are parsed into
  the same source library used by the WebUI; duplicate content is skipped and
  re-uploading a changed file under the same name creates a new version. Each
  file is reported independently, and the command exits nonzero if any upload
  fails. `--json` emits the full per-file report.
- `trash` remains the existing maintenance command for listing or purging
  trashed files. Other file-management commands are not yet available in the
  CLI; parity work continues through the beta.

The CLI and WebUI use the same local project and file records. Use `fts --help`
for the current installed command set.

## Mining jobs

Mining uses the configured **Mining** helper and runs in a persistent worker,
so a queued job continues after the `fts` command exits:

```bash
fts mining start --project "Maintenance notes" "maintenance.txt"
fts mining start --project <project-id> --kind fill maintenance.txt --json
fts mining list --project <project-id> --active
fts mining status <job-id> --project <project-id>
fts mining cancel <job-id> --project <project-id>
```

- `start` accepts parsed file names or source ids. `mine` builds a file's
  inventory and Q&A pairs; `fill` retries open data points on an already-mined
  file. The configured helper must be usable before any job is queued.
- `list` shows job status, stage and file; `--active` limits it to queued/running
  work. `status` and `cancel` require the owning project so a job from another
  project cannot be mistaken for this one. All commands support `--json`.

## Q&A review and export

Inspect and review pairs from the same queue used by the Pairs page:

```bash
fts qa list --project "Maintenance notes" --status pending
fts qa list --project <project-id> --source maintenance.txt --search pump --json
fts qa approve --project <project-id> <pair-id> <another-pair-id>
fts qa reject --project <project-id> <pair-id>
fts qa reject --project <project-id> <pair-id> --yes --json
fts dataset build --project <project-id>
```

- `list` includes project-wide pending/approved/rejected counts and accepts a
  status, source file/id, text search, offset and page limit.
- `approve` and `reject` update the same pair records and review timestamps as
  the WebUI. Rejecting the last pair that covers a mined fact requires a
  terminal confirmation; non-interactive scripts must pass `--yes`.
- Export is `fts dataset build`, not a second Q&A export path: it applies the
  same completeness gate, approved-pair selection, optional grounded context,
  and dataset registration as the WebUI's **Export approved → Training**.

## Project versions

Pin the current project inputs and inspect how snapshots build on each other:

```bash
fts versions list --project "Maintenance notes"
fts versions pin --project <project-id> --label "first reviewed set" --notes "Before the first training run"
fts versions pin --project <project-id> --label "independent experiment" --root
fts versions show <version-id> --project <project-id> --json
fts versions lineage <version-id> --project <project-id>
fts versions delete <version-id> --project <project-id>
```

- `pin` records the project's current datasets, recent training runs and latest
  Smart Brain build, matching Quick Work step 6. It builds on the newest
  version by default; use `--parent <version-id>` to branch from another one or
  `--root` to start a separate lineage.
- `list`, `show`, and `lineage` accept `--json`; IDs shown by `list` identify
  versions for the other commands. Lineage is reported oldest to newest.
- `delete` asks for confirmation in a terminal. Scripts must pass `--yes`.
  It deletes only the snapshot record; datasets, runs, model files and Brain
  indexes referenced by the manifest remain untouched.

## Trained model exports

Queue and inspect exports from a terminal using the same validator, durable job
records, and cancellation behavior as the Export page:

```bash
fts export start <run-id> --project "Maintenance notes" --quants Q4_K_M Q5_K_M
fts export start <run-id> --project <project-id> --format gguf --quants Q4_K_M --imatrix --json
fts export start <run-id> --project <project-id> --quant Q8_0
fts export start <run-id> --project <project-id> --format merged
fts export start <run-id> --project <project-id> --format heretic --force
fts export list --project <project-id> --json
fts export status <export-id> --project <project-id>
fts export cancel <export-id> --project <project-id>
```

- `start` accepts `gguf`, `merged`, `abliterated`, and `heretic`; the optional
  Heretic tool must be installed first. GGUF defaults to Q4_K_M; use `--quants`
  for one or more formats, `--quant` for the single-quant form, and `--imatrix`
  to calibrate quantization from the run's training text. `--base-model` selects
  a compatible base for merge-at-export; `--no-auto-merge` refuses an unmerged
  run; `--force` overwrites existing outputs.
- `start` returns as soon as the persistent job is queued. `list` shows this
  project's export records; `status` shows phase, progress, and the output or
  error; `cancel` requests a cooperative stop. A torch merge stops between
  phases; a running llama.cpp child is stopped as a process group.
- Invalid formats, unavailable tools, missing training data for imatrix,
  project/run mismatches, and a busy GPU/export return a reason and a non-zero
  exit. All commands accept `--json` for machine-readable results.
