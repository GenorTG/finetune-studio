/* ============================================================
   FINETUNE STUDIO — Command Palette (Ctrl/Cmd+K)
   ------------------------------------------------------------
   Center-screen overlay, fuzzy-matched across:
     • Static nav (dashboard / projects / hf / inference)
     • All projects × their 9 sub-pages
     • Recent destinations (localStorage, MRU-style)
   Multi-token query: every whitespace-separated token must
   match somewhere in the haystack (name, group, href).
   ============================================================ */
(function () {
  const PALETTE_OPEN = "fts.palette.open";
  const RECENT_KEY  = "fts.palette.recent";
  const PROJ_CACHE  = "fts.palette.projects";      // cached list
  const PROJ_TTL_MS = 60 * 1000;                   // 1 min freshness

  const $ = (id) => document.getElementById(id);
  const backdrop = $("palette-backdrop");
  const palette  = $("palette");
  const input    = $("palette-input");
  const results  = $("palette-results");
  const trigger  = $("sb-palette");

  /* ── Platform-aware button label ───────────────────────────── */
  const isMac = /Mac|iPod|iPhone|iPad/.test(navigator.platform);
  const MOD_LABEL = isMac ? "⌘" : "Ctrl";
  if (trigger) {
    // Replace inner content with platform-correct label.
    trigger.innerHTML = `<span class="sb-palette-key">${MOD_LABEL}</span><span>K</span>`;
    trigger.title = `Command palette (${MOD_LABEL}+K)`;
  }

  /* ── Static (non-project) nav items ────────────────────────── */
  const STATIC_NAV = [
    { href: "/",                 label: "dashboard",   group: "sys",    keywords: "home overview" },
    { href: "/projects",         label: "projects",    group: "sys",    keywords: "list all" },
    { href: "/models/explore",   label: "hf",          group: "tools",  keywords: "huggingface explorer download" },
    { href: "/inference",        label: "inference",   group: "tools",  keywords: "chat model load llm" },
  ];
  const PROJECT_SUB_PAGES = [
    { label: "overview",    href_suffix: "",            keywords: "home summary" },
    { label: "data",        href_suffix: "/data",       keywords: "files upload" },
    { label: "data-prep",   href_suffix: "/data-prep",  keywords: "preparation cleaning qa" },
    { label: "rag",         href_suffix: "/rag",        keywords: "retrieval index corpus chunks embeddings" },
    { label: "training",    href_suffix: "/training",   keywords: "train finetune lora" },
    { label: "testing",     href_suffix: "/testing",    keywords: "tests eval suite" },
    { label: "benchmarks",  href_suffix: "/benchmarks", keywords: "bench perf stats" },
    { label: "chat",        href_suffix: "/chat",       keywords: "talk conversation" },
    { label: "agentic",     href_suffix: "/agentic",    keywords: "agent tools autonomous" },
  ];

  /* ── Cache for project list ───────────────────────────────── */
  let projectsCache = null;
  let projectsCachedAt = 0;
  async function getProjects() {
    if (projectsCache && (Date.now() - projectsCachedAt) < PROJ_TTL_MS) {
      return projectsCache;
    }
    try {
      const r = await fetch("/api/projects");
      if (!r.ok) throw new Error("HTTP " + r.status);
      projectsCache = await r.json();
      projectsCachedAt = Date.now();
      return projectsCache;
    } catch (e) {
      console.warn("[palette] /api/projects fetch failed:", e);
      return [];
    }
  }

  /* ── Build full index ─────────────────────────────────────── */
  let navIndex = [];
  async function rebuildIndex() {
    const items = [...STATIC_NAV];
    const projects = await getProjects();
    for (const p of projects) {
      for (const sub of PROJECT_SUB_PAGES) {
        items.push({
          href: `/projects/${p.id}${sub.href_suffix}`,
          label: sub.label,
          group: p.name,
          keywords: sub.keywords + " " + p.name + " " + p.id,
          projectId: p.id,
          projectName: p.name,
        });
      }
    }
    navIndex = items;
  }

  /* ── Multi-token fuzzy match ──────────────────────────────── */
  function fuzzyScore(query, item) {
    if (!query) return 1;
    const tokens = query.toLowerCase().split(/\s+/).filter(Boolean);
    if (tokens.length === 0) return 1;

    const haystack = [
      item.label,
      item.group,
      item.href,
      item.keywords || "",
    ].join(" ").toLowerCase();

    let total = 0;
    for (const tok of tokens) {
      if (haystack.includes(tok)) {
        // Word-boundary bonus: match starts a word
        const re = new RegExp("\\b" + tok.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"), "i");
        if (re.test(haystack)) total += 50;
        else total += 20;
      } else {
        // Try subsequence match (fuzzy)
        let ti = 0;
        for (let hi = 0; hi < haystack.length && ti < tok.length; hi++) {
          if (haystack[hi] === tok[ti]) ti++;
        }
        if (ti < tok.length) return 0;  // token didn't match at all
        total += 5;
      }
    }
    // Boost exact matches
    if (item.label.toLowerCase() === query.toLowerCase()) total += 200;
    return total;
  }

  /* ── Recent destinations ──────────────────────────────────── */
  function getRecent() {
    try { return JSON.parse(localStorage.getItem(RECENT_KEY) || "[]"); }
    catch (e) { return []; }
  }
  function pushRecent(href) {
    const recent = getRecent().filter((r) => r !== href);
    recent.unshift(href);
    while (recent.length > 5) recent.pop();
    try { localStorage.setItem(RECENT_KEY, JSON.stringify(recent)); }
    catch (e) {}
  }

  /* ── Render results ───────────────────────────────────────── */
  let currentResults = [];
  let cursor = 0;

  function renderResults(query) {
    let scored;
    if (!query) {
      // Empty: show recents first, then top static
      const recent = getRecent();
      const recentItems = recent
        .map((href) => navIndex.find((it) => it.href === href))
        .filter(Boolean);
      const restStatic = STATIC_NAV.slice(0, 4);
      scored = [
        ...recentItems.map((it) => ({ ...it, _section: "recent" })),
        ...restStatic.map((it) => ({ ...it, _section: "static" })),
      ];
    } else {
      scored = navIndex
        .map((it) => ({ ...it, score: fuzzyScore(query, it) }))
        .filter((it) => it.score > 0)
        .sort((a, b) => b.score - a.score)
        .slice(0, 14);
    }

    currentResults = scored;
    if (cursor >= scored.length) cursor = 0;

    if (scored.length === 0) {
      results.innerHTML =
        `<div class="palette-empty">no matches for "${escapeHtml(query)}"</div>` +
        `<div class="palette-empty-tip">try: <code>qa3 rag</code> · <code>train</code> · <code>hf</code></div>`;
      return;
    }

    // Group by project / group for nicer display
    let html = "";
    let lastGroup = null;
    for (let i = 0; i < scored.length; i++) {
      const it = scored[i];
      const g = it._section === "recent" ? "recent" : it.group;
      if (g !== lastGroup) {
        if (lastGroup !== null) html += "</div>";
        const label = g === "recent" ? "RECENT" : g.toUpperCase();
        html += `<div class="palette-group"><span class="palette-group-label">${escapeHtml(label)}</span>`;
        lastGroup = g;
      }
      html += `
        <button type="button" class="palette-row ${i === cursor ? 'active' : ''}"
                data-href="${escapeHtml(it.href)}" data-idx="${i}">
          <span class="palette-row-icon">${iconFor(it.label)}</span>
          <span class="palette-row-label">${escapeHtml(it.label)}</span>
          <span class="palette-row-href">${escapeHtml(it.href)}</span>
        </button>`;
    }
    if (lastGroup !== null) html += "</div>";
    results.innerHTML = html;
  }

  function iconFor(label) {
    const m = {
      dashboard: "▦", projects: "▤", overview: "◉", data: "▤",
      "data-prep": "▥", rag: "⌗", training: "▣", testing: "✓",
      benchmarks: "▲", chat: "◊", agentic: "✶",
      hf: "◈", inference: "◉",
    };
    return m[label] || "›";
  }
  function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, (c) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  }

  function moveCursor(delta) {
    if (currentResults.length === 0) return;
    cursor = (cursor + delta + currentResults.length) % currentResults.length;
    renderResults(input.value);
  }

  function commitCursor() {
    const r = currentResults[cursor];
    if (r) {
      pushRecent(r.href);
      closePalette();
      if (window.ftsSPA && window.ftsSPA.navigate) window.ftsSPA.navigate(r.href);
      else location.href = r.href;
    }
  }

  async function openPalette() {
    if (!palette) return;
    await rebuildIndex();
    palette.hidden = false;
    backdrop.hidden = false;
    palette.getBoundingClientRect();
    palette.classList.add("open");
    backdrop.classList.add("open");
    input.value = "";
    input.focus();
    cursor = 0;
    renderResults("");
    try { sessionStorage.setItem(PALETTE_OPEN, "1"); } catch (e) {}
  }
  function closePalette() {
    if (!palette) return;
    palette.classList.remove("open");
    backdrop.classList.remove("open");
    setTimeout(() => {
      palette.hidden = true;
      backdrop.hidden = true;
    }, 180);
    try { sessionStorage.removeItem(PALETTE_OPEN); } catch (e) {}
  }

  /* ── Event wiring ─────────────────────────────────────────── */
  if (trigger) trigger.addEventListener("click", openPalette);
  if (backdrop) backdrop.addEventListener("click", closePalette);

  if (input) input.addEventListener("input", () => {
    cursor = 0;
    renderResults(input.value);
  });
  if (input) input.addEventListener("keydown", (ev) => {
    if (ev.key === "ArrowDown") { ev.preventDefault(); moveCursor(1); }
    else if (ev.key === "ArrowUp") { ev.preventDefault(); moveCursor(-1); }
    else if (ev.key === "Enter") { ev.preventDefault(); commitCursor(); }
    else if (ev.key === "Escape") { ev.preventDefault(); closePalette(); }
    else if (ev.key === "Tab") { ev.preventDefault(); moveCursor(ev.shiftKey ? -1 : 1); }
  });

  if (results) results.addEventListener("click", (ev) => {
    const row = ev.target.closest(".palette-row");
    if (!row) return;
    cursor = parseInt(row.dataset.idx || "0", 10);
    commitCursor();
  });

  /* ── Global hotkey: Ctrl/Cmd+K ────────────────────────────── */
  document.addEventListener("keydown", (ev) => {
    // Don't hijack when typing elsewhere
    const t = ev.target;
    const isFormField = t && (t.tagName === "INPUT" || t.tagName === "TEXTAREA" || t.isContentEditable);
    if (isFormField && t !== input) return;

    const mod = isMac ? ev.metaKey : ev.ctrlKey;
    if (mod && ev.key.toLowerCase() === "k") {
      ev.preventDefault();
      if (palette && !palette.hidden) closePalette(); else openPalette();
      return;
    }
    if (ev.key === "Escape" && palette && !palette.hidden) {
      ev.preventDefault();
      closePalette();
    }
  });

  // Eagerly preload projects so palette opens instantly
  getProjects().catch(() => {});

  // Expose for SPA swaps
  window.ftsPalette = {
    open: openPalette, close: closePalette,
    refresh: async () => { projectsCache = null; await rebuildIndex(); },
  };
})();
