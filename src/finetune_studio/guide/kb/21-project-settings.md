---
id: project-settings
title: Project settings and the WebUI log
page: project_settings
keywords: project settings name base model description tags notes system prompt save export archive project tar.gz webui log tail service log refresh auto refresh settings
---
## Purpose
Change settings for this project only: its name, base model, description, tags, notes, and system prompt. The folded WebUI log tail below the form is the studio service log shared across projects.

## Workflow
1. Edit the project fields and choose **Save** to store them. These are not the app-wide AI assistant, compute-device, access, or network settings; those live under the top-level **Settings** page.
2. **Export** downloads this project as a `.tar.gz` archive. **Back to overview** returns to the project home.
3. Expand **WebUI log tail** to fetch the recent service log. **Refresh** reloads it once; **Auto-refresh** polls every five seconds until switched off or you leave the page.

## Key controls
- `#settings-form` — the project name, base model, description, tags, notes, and system prompt.
- `#log-tail-card` — the folded, shared WebUI service log.
- `#log-refresh-btn` — fetch the log now.
- `#log-auto-btn` — toggle five-second refresh.

## What to check
The form shows only project-level settings. The log badge reports live, stale/unavailable, or offline status; its timestamp and lines refresh when the panel opens or you press **Refresh**. The activity drawer is a separate view of recent jobs and events.

## Common mistakes
Changing this project's base model does not change the app-wide assistant or compute device. The log is shared service output, not a private per-project log; the page folds it until opened.
