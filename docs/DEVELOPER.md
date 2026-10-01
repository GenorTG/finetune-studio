# Developer documentation index

This is the engineering reference for contributors and maintainers. It is
separate from the user-facing [README](../README.md), [installation guide](INSTALL.md),
and [tutorial](TUTORIAL.md). Keep it on the developer documentation branch;
branch protection does not make a public repository private.

## Start here

- [`../AGENTS.md`](../AGENTS.md): repo workflow, commands, deployment constraints,
  and load-bearing invariants.
- [`../HANDOFF.md`](../HANDOFF.md): verified current state, next steps, blockers.
- [`WORKPLAN.md`](WORKPLAN.md): project task ordering and durable operating plan.
- [`ARCHITECTURE.md`](ARCHITECTURE.md): application layers, routes, persistence,
  and system data flow.
- [`CODEMAP.md`](CODEMAP.md): generated Python symbol/import index; regenerate with
  `make codemap` after adding or moving symbols.

## Implementation maps

- [`modules/webui-frontend.md`](modules/webui-frontend.md): CSS, JavaScript,
  templates, SPA lifecycle, and frontend invariants.
- [`modules/`](modules/): audited backend maps for CLI, DB, data prep, parsers,
  file storage, RAG, training, benchmarks, testing, and WebUI routes.
- [`DEPLOYMENT.md`](DEPLOYMENT.md): service, update, and host operations.
- [`DEPENDENCIES.md`](DEPENDENCIES.md): dependency, CUDA, and platform notes.

## Audit boundary

Treat each map as a navigation aid, not a substitute for checking the current
implementation. The repository audit records exact file/read coverage in
`HANDOFF.md`; binary test fixtures are content assets, not line-readable source.
