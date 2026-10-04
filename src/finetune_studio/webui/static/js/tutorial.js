/* ============================================================
   FINETUNE STUDIO — First-run onboarding tutorial
   ------------------------------------------------------------
   A short, task-oriented tour: get a base model → create a
   project → Quick work → watch long jobs → check results →
   shortcuts. Steps can show live state ("you have 0 base
   models") and a call-to-action button that does the next
   thing. The dialog sits next to the highlighted element so
   the page stays visible.

   Triggered on first visit (localStorage flag) or manually via
   window.ftsTutorial.start({ force: true }) (Settings → Replay).
   The current step survives a full page reload (sessionStorage),
   so a non-SPA navigation resumes instead of restarting.
   ============================================================ */
(function () {
  const SEEN_KEY = "fts.tutorial.seen";
  const SEEN_AT_KEY = "fts.tutorial.seenAt";
  const STEP_KEY = "fts.tutorial.step";  // sessionStorage: active tour position

  async function getJSON(url) {
    const r = await fetch(url, { cache: "no-store" });
    if (!r.ok) throw new Error("HTTP " + r.status);
    return r.json();
  }
  async function trainableModels() {
    const d = await getJSON("/api/models/list?for_training=true");
    return Array.isArray(d) ? d : [];
  }
  async function projectList() {
    const d = await getJSON("/api/projects");
    return Array.isArray(d) ? d : (d && Array.isArray(d.projects) ? d.projects : []);
  }

  // Step fields:
  //   target  css selector to highlight (null = centered dialog)
  //   route   path to navigate to before showing the step
  //   title, body
  //   live    async () => {text, ok}   current state, shown under the body
  //   cta     async () => {label, href} | null   button that does the next thing
  const STEPS = [
    {
      target: null,
      title: "WELCOME TO FINETUNE STUDIO",
      body: "Turn your own documents into an AI that knows them, on your own GPU. " +
            "Two ways: TRAIN a model on question-answer pairs made from your files, " +
            "or SEARCH your files from a chat (RAG) with no training at all. " +
            "This 1-minute tour walks the path from nothing to a working model. " +
            "Esc skips; Settings → Replay brings it back.",
    },
    {
      target: ".sb-group-tools",
      title: "1 · GET A BASE MODEL",
      body: "Training starts from an existing open model. Model library downloads " +
            "one from Hugging Face; My models lists what is already on this machine. " +
            "For a first run pick something small (Qwen3-0.6B, ~1.2 GB, minutes to train); " +
            "for real results a 4B model on a 24 GB GPU.",
      live: async () => {
        const ms = await trainableModels();
        if (!ms.length) return { ok: false, text: "You have no trainable base model yet — download one first." };
        const names = ms.slice(0, 3).map((m) => m.name).join(", ");
        return { ok: true, text: `✓ ${ms.length} base model(s) ready: ${names}${ms.length > 3 ? ", …" : ""}` };
      },
      cta: async () => ((await trainableModels()).length ? null
        : { label: "Open Model library", href: "/models/explore" }),
    },
    {
      target: "#new-project-btn",
      route: "/projects",
      title: "2 · CREATE A PROJECT",
      body: "A project holds one job's files, datasets, training runs, tests and " +
            "exports, kept apart from everything else. Name it after what the model " +
            "should know (\"Support FAQ\", \"Product manuals\").",
      live: async () => {
        const ps = await projectList();
        return ps.length ? { ok: true, text: `✓ You have ${ps.length} project(s).` }
                         : { ok: false, text: "No projects yet." };
      },
    },
    {
      target: null,
      title: "3 · QUICK WORK DOES THE CHAIN",
      body: "Inside a project, ⚡ Quick work runs everything on one page: " +
            "① upload files (PDF, DOCX, TXT, …) → ② generate question-answer pairs from them " +
            "→ ③ build a dataset → ④ train → ⑤ test on questions held back from training " +
            "→ ⑥ export or chat. Each step shows a status and links to its full page when " +
            "you want more control (Files, Pairs, Training, …).",
    },
    {
      target: "#sb-activity",
      title: "4 · LONG JOBS RUN IN THE BACKGROUND",
      body: "Downloads, pair generation and training keep running while you browse. " +
            "This button opens the activity feed with live progress; project pages also " +
            "show GPU and RAM use. A small model trains in minutes; a 4B model can take an hour.",
    },
    {
      target: null,
      title: "5 · CHECK THE RESULT YOURSELF",
      body: "Testing scores the trained model on held-out questions. Benchmarks keeps every " +
            "result: open one, read each answer, and override the judge where it got it wrong — " +
            "automatic scores are a heuristic, so spot-check 10–20 answers before trusting one. " +
            "Then Export (GGUF for llama.cpp, LM Studio, Ollama) or talk to it in Chat.",
    },
    {
      target: "#sb-palette",
      title: "SHORTCUT: Ctrl+K",
      body: "Press Ctrl+K (Cmd+K on Mac) anywhere to jump to any page of any project — " +
            "type \"train\", \"export\" or a project name.",
      cta: async () => ((await projectList()).length
        ? null : { label: "Create my first project", href: "/projects#new" }),
      finish: true,
    },
  ];

  let currentStep = 0;
  let overlay, onNavHandler, renderSeq = 0;

  function el(id) { return document.getElementById(id); }

  function init() {
    if (el("tutorial-overlay")) { overlay = el("tutorial-overlay"); return; }
    overlay = document.createElement("div");
    overlay.id = "tutorial-overlay";
    overlay.className = "tutorial-overlay";
    overlay.hidden = true;
    overlay.innerHTML = `
      <div class="tutorial-highlight" id="tutorial-highlight" hidden></div>
      <div class="tutorial-dialog" id="tutorial-dialog" role="dialog" aria-modal="true"
           aria-labelledby="tutorial-title" aria-describedby="tutorial-body">
        <div class="tutorial-head">
          <span class="tutorial-step-count" id="tutorial-step-count"></span>
          <button class="tutorial-skip" id="tutorial-skip" title="Close the tour (Esc)">SKIP</button>
        </div>
        <div class="tutorial-body">
          <h2 class="tutorial-title" id="tutorial-title"></h2>
          <div class="tutorial-text" id="tutorial-body"></div>
          <div class="tutorial-live" id="tutorial-live" hidden></div>
        </div>
        <div class="tutorial-foot">
          <div class="tutorial-dots" id="tutorial-dots"></div>
          <div class="tutorial-nav">
            <button class="tutorial-btn" id="tutorial-cta" hidden></button>
            <button class="tutorial-btn" id="tutorial-prev">← BACK</button>
            <button class="tutorial-btn primary" id="tutorial-next">NEXT →</button>
          </div>
        </div>
      </div>
    `;
    document.body.appendChild(overlay);

    el("tutorial-skip").addEventListener("click", finish);
    el("tutorial-prev").addEventListener("click", prev);
    el("tutorial-next").addEventListener("click", next);
    el("tutorial-cta").addEventListener("click", () => {
      const href = el("tutorial-cta").dataset.href;
      finish();
      if (href) go(href);
    });
    document.addEventListener("keydown", (ev) => {
      if (!overlay || overlay.hidden) return;
      if (ev.key === "Escape") { ev.preventDefault(); finish(); }
      else if (ev.key === "ArrowRight") { ev.preventDefault(); next(); }
      else if (ev.key === "ArrowLeft") { ev.preventDefault(); prev(); }
    });
    window.addEventListener("resize", () => { if (overlay && !overlay.hidden) place(STEPS[currentStep]); });
  }

  function go(href) {
    if (window.ftsSPA && window.ftsSPA.navigate) window.ftsSPA.navigate(href);
    else location.href = href;
  }

  async function start({ force = false } = {}) {
    init();
    const resume = readStep();
    if (!force && hasSeen() && resume === null) return;
    currentStep = force || resume === null ? 0 : resume;
    overlay.style.display = "";  // finish() forces display:none; undo on replay
    overlay.hidden = false;
    if (!onNavHandler) {
      // The tour can span pages: re-render once the new page has settled.
      onNavHandler = () => { if (overlay && !overlay.hidden) setTimeout(renderStep, 300); };
      document.addEventListener("fts:navigated", onNavHandler);
    }
    await renderStep();
  }

  // Put the dialog beside the highlighted element (below it, or above when
  // there is no room); centered when the step has no target.
  function place(step) {
    const dialog = el("tutorial-dialog");
    const highlight = el("tutorial-highlight");
    const target = step.target && document.querySelector(step.target);
    if (!target || target.getBoundingClientRect().width === 0) {
      highlight.hidden = true;
      dialog.classList.remove("anchored");
      dialog.style.top = dialog.style.left = "";
      return;
    }
    const r = target.getBoundingClientRect();
    const pad = 6;
    Object.assign(highlight.style, {
      left: (r.left - pad) + "px", top: (r.top - pad) + "px",
      width: (r.width + pad * 2) + "px", height: (r.height + pad * 2) + "px",
    });
    highlight.hidden = false;
    dialog.classList.add("anchored");
    const dw = dialog.offsetWidth, dh = dialog.offsetHeight, gap = 16;
    const below = r.bottom + gap;
    const top = below + dh <= window.innerHeight - 8 ? below : Math.max(8, r.top - gap - dh);
    const left = Math.min(Math.max(8, r.left), window.innerWidth - dw - 8);
    dialog.style.top = top + "px";
    dialog.style.left = left + "px";
  }

  async function renderStep() {
    const seq = ++renderSeq;
    const step = STEPS[currentStep];
    saveStep(currentStep);
    if (step.route && location.pathname !== step.route) {
      go(step.route);  // fts:navigated (or the reload resume) renders this step
      return;
    }
    el("tutorial-title").textContent = step.title;
    el("tutorial-body").textContent = step.body;
    el("tutorial-step-count").textContent = `STEP ${currentStep + 1} / ${STEPS.length}`;
    el("tutorial-next").textContent = step.finish ? "FINISH ✓" : "NEXT →";
    el("tutorial-prev").disabled = currentStep === 0;
    el("tutorial-dots").innerHTML = STEPS.map((_, i) =>
      `<span class="tutorial-dot ${i === currentStep ? "active" : ""}"></span>`).join("");

    const live = el("tutorial-live");
    const cta = el("tutorial-cta");
    live.hidden = true; cta.hidden = true;

    const target = step.target && document.querySelector(step.target);
    if (target) {
      target.scrollIntoView({ behavior: "smooth", block: "center" });
      await new Promise((r) => setTimeout(r, 250));
    }
    if (seq !== renderSeq) return;
    place(step);

    // Live state + call-to-action load after the dialog is visible.
    try {
      const [state, action] = await Promise.all([
        step.live ? step.live() : null,
        step.cta ? step.cta() : null,
      ]);
      if (seq !== renderSeq) return;
      if (state) {
        live.textContent = state.text;
        live.classList.toggle("ok", !!state.ok);
        live.hidden = false;
      }
      if (action) {
        cta.textContent = action.label;
        cta.dataset.href = action.href;
        cta.hidden = false;
      }
      place(step);  // dialog height may have changed
    } catch (_) { /* state is a nicety; the tour works without it */ }
  }

  function next() {
    if (currentStep < STEPS.length - 1) { currentStep++; renderStep(); }
    else finish();
  }

  function prev() {
    if (currentStep > 0) { currentStep--; renderStep(); }
  }

  function finish() {
    if (overlay) {
      overlay.hidden = true;
      // .tutorial-overlay is position:fixed and can override [hidden]{display:none}.
      overlay.style.display = "none";
    }
    const highlight = el("tutorial-highlight");
    if (highlight) highlight.hidden = true;
    if (onNavHandler) {
      document.removeEventListener("fts:navigated", onNavHandler);
      onNavHandler = null;
    }
    renderSeq++;
    clearStep();
    markSeen();
  }

  function readStep() {
    try {
      const v = sessionStorage.getItem(STEP_KEY);
      const n = v === null ? NaN : parseInt(v, 10);
      return Number.isInteger(n) && n >= 0 && n < STEPS.length ? n : null;
    } catch (e) { return null; }
  }
  function saveStep(n) { try { sessionStorage.setItem(STEP_KEY, String(n)); } catch (e) {} }
  function clearStep() { try { sessionStorage.removeItem(STEP_KEY); } catch (e) {} }

  function hasSeen() {
    try { return localStorage.getItem(SEEN_KEY) === "1"; }
    catch (e) { return false; }
  }

  function markSeen() {
    try {
      localStorage.setItem(SEEN_KEY, "1");
      localStorage.setItem(SEEN_AT_KEY, new Date().toISOString());
    } catch (e) {}
  }

  function resetSeen() {
    try {
      localStorage.removeItem(SEEN_KEY);
      localStorage.removeItem(SEEN_AT_KEY);
    } catch (e) {}
    clearStep();
  }

  window.ftsTutorial = { start, finish, resetSeen, hasSeen, markSeen, steps: STEPS.length };

  // Auto-trigger on first visit (or resume after a reload), after paint.
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", () => setTimeout(() => start(), 1500));
  } else {
    setTimeout(() => start(), 1500);
  }
})();
