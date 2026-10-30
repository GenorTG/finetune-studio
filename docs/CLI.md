# Command-line reference

The `fts` command operates on the same local project database as the WebUI.
Start with `fts --help` to see every installed command and `fts <command> --help`
for that command's options. Commands that run locally do not require the WebUI
server unless their help says otherwise.

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
