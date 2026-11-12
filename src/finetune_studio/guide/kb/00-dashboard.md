---
id: dashboard
title: Dashboard
page: dashboard
keywords: dashboard home system overview host resources projects local models training gpu quick actions create project download model inference tour
---
## Purpose
The Dashboard is the studio-wide start page. It summarizes project and local-model counts, whether a model is loaded, live training status and progress, GPU memory/use, and recent projects. It is not an individual project's home page.

## Workflow
1. Check Host Resources and the Projects, Local models, Training, and GPU cards for the current studio state.
2. Use Quick actions to create a project, reopen the last one, go to its Package page, or replay the tour (model downloads and local chat are linked below them).
3. Open a project card or **All projects** to enter project-specific files, training, Smart Brain, tests, and exports.

## Key controls
- `#dash-actions` — Quick actions: **＋ New project**, **Open <your last project>** (its Smart Brain page), **Package it** (that project's Package page) and **Take the tour**. Downloading a model and loading one to chat are linked below them: training and local chat are optional.
- `#dash-open-last` — opens the project you worked on last.
- `#dash-package` — that project's Package page.
- `#dash-training-status` — current training state, refreshed every three seconds.
- `#dash-training-progress` — current training progress, refreshed every three seconds.
- `#dash-gpu` — GPU memory summary, refreshed every five seconds.
- `#dash-gpu-sub` — GPU details, refreshed every five seconds.

## What to check
The Dashboard reports studio-wide counts and the active GPU/model/training state. If no project exists, the Projects card offers a first-project link. Open a project to see its own workflow and recent activity.

## Common mistakes
The Dashboard is not a project's overview and does not contain that project's next-step checklist. **Take the tour** replays the onboarding walkthrough; it does not create or change project data.
