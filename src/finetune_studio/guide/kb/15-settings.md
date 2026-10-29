---
id: settings
title: Settings page and compute device
page: settings
keywords: settings update access password sign in login token api token network address port host cors proxy judge test judge auto-judge helper assistant ai assistant connection connections api provider key opencode openai openrouter jobs instructions vision pictures compute device cpu gpu cuda restart tutorial paths storage debug heretic abliteration uncensored decensor install upstream
---
## Purpose
Global settings: updates, tutorial replay, access (password, API tokens, network address), the AI assistant, the test judge, the compute device, debug info and storage paths.

## Workflow
1. **Updates:** **Check** asks GitHub whether this install's branch has a newer version (and lists what changed), then checks the Python environment, GPU driver and llama.cpp tools; it changes nothing. **Apply update** downloads it, updates the packages, migrates the database and restarts (under a minute; with a job running the restart is skipped until later). **Repair** rebuilds the Python environment from scratch (10–20 minutes): only when Check reports a broken environment.
2. **Compute device:** choose which device the app uses at the **next start** (Auto = best detected GPU, CPU only if none). The choice is saved to a file and applied only at process start; explicit `FTS_GPU_*`, `FTS_DEVICE` or `*_VISIBLE_DEVICES` environment settings override it and show as "overridden". CPU-only masks every card.
3. **AI assistant:** one assistant does all the model work. *Connections* lists the local model on this machine plus any API connections you add (**Add API connection**: name, provider preset, URL, key, model, reasoning effort, parallel calls; **Test connection** makes one real call). The presets in this form are **OpenAI**, **OpenRouter**, **OpenCode Go**, and **Custom OpenAI-compat URL**. To use Claude through OpenRouter, choose **OpenRouter**; its default model is `anthropic/claude-3.5-sonnet`, which you can change to another OpenRouter model. This form does not have an Anthropic-specific preset. Keys are stored encrypted on this computer, sent only to their own URL and never shown again. **Main assistant** picks the connection every job uses. In **Jobs**, each job (Guide chat, mining, study sets, reworded quizzes, preference pairs, test judge, use-case teacher, reading pictures) can run on another connection and can get **extra instructions** that are added in front of its own prompt. *Reading pictures* is off until you pick a connection that accepts images; **Test reading pictures** checks it. The *Sends* column says what leaves the machine when a job runs on an API.
4. **Heretic (integrated optional feature):** the app exposes refusal-removal exports using Heretic's parameter search, rather than one fixed abliteration strength. The computation runs in a separately licensed program (AGPL-3.0), which Settings installs and manages in `~/.finetune-studio/heretic`; the app reuses its own PyTorch, so this costs a few hundred MB rather than another copy of CUDA. The Heretic format remains an Export-page feature. **Install / update** downloads or repairs that separate environment; **Check upstream** asks GitHub whether it is behind; **Remove** deletes only that tool environment, and makes the Heretic export unavailable until it is installed again.
5. **Test judge:** whether a finished test run starts judging by itself (off by default). The judge model is the *Test judge* job above (the Testing page can override it per run).
6. **Access & network:** who can open the web UI. Default: this computer only (127.0.0.1), no password.
   To use it from other devices: set a password first, then choose "every network" and restart the service.
   API tokens let scripts call the API (`Authorization: Bearer <token>`); a token is shown once.
   Forgot the password: on the machine itself run `fts auth set-password`.

## Key controls
- `#btn-update-check` — look for a newer version and check the install (changes nothing).
- `#btn-update-apply` — install the newer version and restart.
- `#compute-select` — compute device for the next start.
- `#btn-compute-save` — save the compute device (restart required).
- `#assistant-card` — the AI assistant card.
- `#assistant-main` — the main assistant.
- `#connections-table` — connections: the local model and API connections (Edit, Test, Remove).
- `#btn-conn-add` — add an API connection.
- `#assistant-jobs` — the jobs, the connection each runs on, and their extra instructions.
- `#judge-auto` — judge automatically when a test run finishes.
- `#access-card` — password, listen address, API tokens, signed-in browsers.
- `#heretic-card` — the optional Heretic tool (install, check upstream, remove).
- `#btn-heretic-install` — install or re-install Heretic (reuses this app's PyTorch).
- `#btn-heretic-check` — compare the installed Heretic with its GitHub upstream.
- `#btn-heretic-remove` — delete Heretic's checkout and environment.
- `#access-pw-new` — set or change the password.
- `#btn-replay-tutorial` — replay the tutorial.

## What to check
After saving a compute device, restart the service; the card shows the effective device.

## Common mistakes
Expecting a compute-device change or a new listen address to apply without a restart.
Opening the instance to the network without a password (every page then shows a red warning).
Removing the main assistant's connection or key (refused: pick another main assistant first). Choosing a local main assistant for reading pictures when its GGUF has no mmproj next to it (refused with the reason).
The separate Heretic package/license does not make it a detached product: Settings manages it and Export exposes its format. The Anthropic API is not a preset here; use the OpenRouter preset for Claude.
