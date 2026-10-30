# Command-line reference

The `fts` command operates on the same local project database as the WebUI.
Start with `fts --help` to see every installed command and `fts <command> --help`
for that command's options. Commands that run locally do not require the WebUI
server unless their help says otherwise.

## Projects

List, inspect, and create projects from a terminal:

```bash
fts projects list
fts projects list --json
fts projects show "Project name"
fts projects show <project-id> --json
fts projects create --name "Maintenance notes" \
  --description "Equipment manuals and maintenance procedures" \
  --base-model /path/to/base-model
```

- `list` shows each project's name, id, and base model. With `--json`, it
  also includes the last-updated timestamp.
- `show` accepts a project id or name and prints its description, base model,
  and system prompt. Names match case-insensitively; if more than one project
  matches, use its id.
- `create` requires a non-empty `--name`; description, base model, and system
  prompt are optional. Existing names are rejected case-insensitively unless
  `--allow-duplicate` is provided.
- All three commands accept `--json` for machine-readable output.

The same records appear on the WebUI's **Projects** page. Project data remains
in the configured local Finetune Studio data directory.

This reference documents the project commands currently available. More CLI
parity work is tracked through the beta; use `fts --help` for the current
installed command set.
