/* ============================================================
   FINETUNE STUDIO — First-run onboarding tutorial
   ------------------------------------------------------------
   8 steps. Highlights DOM elements with a glowing border,
   shows a side-panel dialog with skip/next/finish controls.
   Triggered automatically on first visit (localStorage flag)
   or manually via window.ftsTutorial.start().
   ============================================================ */
(function () {
  const SEEN_KEY = "fts.tutorial.seen";
  const SEEN_AT_KEY = "fts.tutorial.seenAt";

  // Each step is: { target: css-selector OR null (centered),
  //                  title, body, route? (optional nav before),
  //                  align: 'left'|'right'|'top'|'bottom'|'center' }
  const STEPS = [
    {
      target: null,
      align: "center",
      title: "WELCOME, HACKER",
      body: "Finetune Studio is a self-hosted LLM workshop: build RAG corpora, " +
            "fine-tune models on your own GPU, run benchmarks, and chat — all " +
            "without leaving your browser. This 8-step tour will show you the ropes.",
    },
    {
      target: ".session-bar",
      align: "bottom",
      title: "THE SESSION BAR",
      body: "Your command center. Three groups of tabs — [SYS], [PROJECT], [TOOLS] — " +
            "let you hop between any page in one click. Active tab has a green notch " +
            "and blinking cursor so you always know where you are.",
    },
    {
      target: "#sb-palette",
      align: "bottom",
      title: "COMMAND PALETTE (Ctrl+K)",
      body: "Don't want to hunt for a tab? Press Ctrl+K (or Cmd+K on Mac) to open " +
            "the command palette. Type to fuzzy-search across every page in every " +
            "project. Try 'qa3 rag' or just 'train' — instant jump.",
    },
    {
      target: null,
      align: "center",
      route: "/",
      title: "DASHBOARD",
      body: "The home view. System status, your projects, and a quick way to " +
            "create a new one. Everything in Finetune Studio happens inside a " +
            "project — projects are how you keep experiments isolated.",
    },
    {
      target: null,
      align: "center",
      route: "/projects",
      title: "PROJECTS",
      body: "Each project bundles its own data, training runs, RAG corpora, " +
            "chat sessions, and benchmarks. You can have many side-by-side " +
            "without them interfering.",
    },
    {
      target: null,
      align: "center",
      route: "/models/explore",
      title: "HF EXPLORER",
      body: "Browse and download models from HuggingFace right from the studio. " +
            "Quantized GGUF models (for llama.cpp) and full safetensors (for " +
            "transformers + LoRA training). All cached locally in ~/.cache.",
    },
    {
      target: null,
      align: "center",
      route: "/inference",
      title: "INFERENCE",
      body: "Chat with any model you've loaded. Pick a model from the dropdown, " +
            "hit Load, and start talking. The sprite on the left pulses its mouth " +
            "every time a token streams in — your AI is alive.",
    },
    {
      target: null,
      align: "center",
      route: "/",
      title: "YOU'RE READY",
      body: "That's the tour. From here: create a project → upload data → " +
            "build a RAG corpus or fine-tune a model → chat or benchmark the " +
            "result. You can replay this tour anytime from Settings.",
      finish: true,
    },
  ];

  let currentStep = 0;
  let overlay, dialog, highlight, onNavHandler;

  function init() {
    // Already injected? Bail.
    if (document.getElementById("tutorial-overlay")) return;

    overlay = document.createElement("div");
    overlay.id = "tutorial-overlay";
    overlay.className = "tutorial-overlay";
    overlay.hidden = true;
    overlay.innerHTML = `
      <div class="tutorial-highlight" id="tutorial-highlight" hidden></div>
      <div class="tutorial-dialog" id="tutorial-dialog" role="dialog"
           aria-labelledby="tutorial-title" aria-describedby="tutorial-body">
        <div class="tutorial-head">
          <span class="tutorial-step-count" id="tutorial-step-count"></span>
          <button class="tutorial-skip" id="tutorial-skip">SKIP</button>
        </div>
        <div class="tutorial-body">
          <h2 class="tutorial-title" id="tutorial-title"></h2>
          <div class="tutorial-text" id="tutorial-body"></div>
        </div>
        <div class="tutorial-foot">
          <div class="tutorial-dots" id="tutorial-dots"></div>
          <div class="tutorial-nav">
            <button class="tutorial-btn" id="tutorial-prev">← BACK</button>
            <button class="tutorial-btn primary" id="tutorial-next">NEXT →</button>
          </div>
        </div>
      </div>
    `;
    document.body.appendChild(overlay);

    document.getElementById("tutorial-skip").addEventListener("click", finish);
    document.getElementById("tutorial-prev").addEventListener("click", prev);
    document.getElementById("tutorial-next").addEventListener("click", next);

    document.addEventListener("keydown", (ev) => {
      if (overlay && !overlay.hidden) {
        if (ev.key === "Escape") { ev.preventDefault(); finish(); }
        else if (ev.key === "ArrowRight") { ev.preventDefault(); next(); }
        else if (ev.key === "ArrowLeft") { ev.preventDefault(); prev(); }
      }
    });
  }

  async function start({ force = false } = {}) {
    init();
    if (!force && hasSeen()) return;

    currentStep = 0;
    overlay.hidden = false;
    await renderStep();

    // Re-render on route changes (the tutorial can span multiple pages).
    if (!onNavHandler) {
      onNavHandler = () => {
        if (overlay && !overlay.hidden) {
          // Wait for new page content to settle
          setTimeout(renderStep, 300);
        }
      };
      document.addEventListener("fts:navigated", onNavHandler);
    }
  }

  async function renderStep() {
    const step = STEPS[currentStep];
    document.getElementById("tutorial-title").textContent = step.title;
    document.getElementById("tutorial-body").textContent = step.body;
    document.getElementById("tutorial-step-count").textContent =
      `STEP ${currentStep + 1} / ${STEPS.length}`;

    // Nav button text
    const nextBtn = document.getElementById("tutorial-next");
    nextBtn.textContent = step.finish ? "FINISH ✓" : "NEXT →";

    // Dots
    const dots = document.getElementById("tutorial-dots");
    dots.innerHTML = STEPS.map((_, i) =>
      `<span class="tutorial-dot ${i === currentStep ? 'active' : ''}"></span>`
    ).join("");

    // Highlight target (if any)
    const highlight = document.getElementById("tutorial-highlight");
    if (step.target) {
      const el = document.querySelector(step.target);
      if (el) {
        el.scrollIntoView({ behavior: "smooth", block: "center" });
        await new Promise((r) => setTimeout(r, 250));
        const r = el.getBoundingClientRect();
        const pad = 6;
        highlight.style.left = (r.left - pad) + "px";
        highlight.style.top = (r.top - pad) + "px";
        highlight.style.width = (r.width + pad * 2) + "px";
        highlight.style.height = (r.height + pad * 2) + "px";
        highlight.hidden = false;
      } else {
        highlight.hidden = true;
      }
    } else {
      highlight.hidden = true;
    }

    // Navigate if step has a route and we're not there yet
    if (step.route && location.pathname !== step.route) {
      if (window.ftsSPA && window.ftsSPA.navigate) {
        window.ftsSPA.navigate(step.route);
      } else {
        location.href = step.route;
      }
    }
  }

  function next() {
    if (currentStep < STEPS.length - 1) {
      currentStep++;
      renderStep();
    } else {
      finish();
    }
  }

  function prev() {
    if (currentStep > 0) {
      currentStep--;
      renderStep();
    }
  }

  function finish() {
    if (overlay) overlay.hidden = true;
    const highlight = document.getElementById("tutorial-highlight");
    if (highlight) highlight.hidden = true;
    markSeen();
  }

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
  }

  // Expose
  window.ftsTutorial = { start, finish, resetSeen, hasSeen, markSeen };

  // Auto-trigger on first visit, after paint
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", () => setTimeout(() => start(), 1500));
  } else {
    setTimeout(() => start(), 1500);
  }
})();
