---
id: projects
title: Projects page
page: projects
keywords: project create new import delete template blank chat training rag base model archive search sort
---
## Purpose
A project groups files, Q&A pairs, datasets, training runs, RAG corpora and chat history. Everything else in the app is scoped to the active project.

## Workflow
1. Click **＋ New project**, choose a template (blank, chat-focused, training, RAG), name it, optionally pick a base model, create.
2. Open the project card to land on its overview; the top strip then shows the project's pages.
3. Projects can be imported from a `.tar.gz`/`.tar` export and bulk-selected for deletion.

## Key controls
- `#new-project-btn` — opens the New project card.
- `#np-form` — the creation form (name, description, base model).
- `#np-base` — optional base model (can be picked later; training has its own base-model dropdown).
- `#proj-search` — filter the list.
- `#import-file` — pick an exported project archive to import.

## What to check
The project id (8 hex chars) shown on the card is what every URL and API uses. Deleting a project also removes its outputs on disk.

## Common mistakes
Using a project you care about for experiments — create a throwaway one instead.
