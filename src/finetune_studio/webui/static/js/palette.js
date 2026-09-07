/* ============================================================
   FINETUNE STUDIO — Command Palette (Ctrl+K)
   ------------------------------------------------------------
   Center-screen overlay, fuzzy-matched list, keyboard nav.
   Loads nav targets from existing <a data-link> elements so
   we never duplicate the route registry.
   ============================================================ */
(function () {
  const PALETTE = "fts.palette.open";

  function $(id) { return document.getElementById(id); }

  const backdrop = $("palette-backdrop");
  const palette = $("palette");
  const input = $("palette-input");
  const results = $("palette-results");
  const trigger = $("sb-palette");

  /* ── Build nav index from rendered DOM ──────────────────────── */
  function indexNav() {
    const items = [];
    document.querySelectorAll(".sb-tab").forEach((a) => {
      const href = a.getAttribute("href");
      if (!href || href.startsWith("http")) return;
      const label = a.querySelector(".sb-tab-label")?.textContent?.trim();
      if (!label) return;
      // Group: which [group] label precedes this tab?
      let group = "";
      let prev = a.previousElementSibling;
      while (prev && !prev.classList.contains("sb-group-label")) {
        prev = prev.previousElementSibling;
      }
      if (prev) group = prev.textContent.trim();
      items.push({ href, label, group, el: a });
    });
    return items;
  }

  let navIndex = indexNav();
  let currentResults = [];
  let cursor = 0;

  function openPalette() {
    if (!palette) return;
    navIndex = indexNav();
    palette.hidden = false;
    backdrop.hidden = false;
    palette.getBoundingClientRect();
    palette.classList.add("open");
    backdrop.classList.add("open");
    input.value = "";
    input.focus();
    cursor = 0;
    renderResults("");
    try { sessionStorage.setItem(PALETTE, "1"); } catch (e) {}
  }
  function closePalette() {
    if (!palette) return;
    palette.classList.remove("open");
    backdrop.classList.remove("open");
    setTimeout(() => {
      palette.hidden = true;
      backdrop.hidden = true;
    }, 180);
    try { sessionStorage.removeItem(PALETTE); } catch (e) {}
  }

  /* ── Simple fuzzy match ─────────────────────────────────────── */
  function fuzzyScore(query, str) {
    if (!query) return 1;
    const q = query.toLowerCase();
    const s = str.toLowerCase();
    if (s === q) return 1000;
    if (s.startsWith(q)) return 500 + (100 - s.length);
    if (s.includes(q)) return 200 - (s.indexOf(q));
    // Subsequence match
    let qi = 0;
    let score = 0;
    for (let si = 0; si < s.length && qi < q.length; si++) {
      if (s[si] === q[qi]) { qi++; score += 1; }
    }
    return qi === q.length ? score : 0;
  }

  function renderResults(query) {
    const scored = navIndex
      .map((item) => ({ ...item, score: fuzzyScore(query, item.label) }))
      .filter((item) => item.score > 0)
      .sort((a, b) => b.score - a.score)
      .slice(0, 12);

    currentResults = scored;
    if (cursor >= scored.length) cursor = 0;

    if (scored.length === 0) {
      results.innerHTML = '<div class="palette-empty">no matches</div>';
      return;
    }

    results.innerHTML = scored.map((item, i) => `
      <button type="button" class="palette-row ${i === cursor ? 'active' : ''}" data-href="${item.href}" data-idx="${i}">
        <span class="palette-row-group">${item.group}</span>
        <span class="palette-row-label">${item.label}</span>
        <span class="palette-row-href">${item.href}</span>
      </button>
    `).join("");
  }

  function moveCursor(delta) {
    if (currentResults.length === 0) return;
    cursor = (cursor + delta + currentResults.length) % currentResults.length;
    renderResults(input.value);
  }

  function commitCursor() {
    const r = currentResults[cursor];
    if (r) {
      closePalette();
      // SPA navigate if available, else hard nav
      if (window.ftsSPA && window.ftsSPA.navigate) window.ftsSPA.navigate(r.href);
      else location.href = r.href;
    }
  }

  /* ── Event wiring ───────────────────────────────────────────── */
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
  });

  // Click on a row
  if (results) results.addEventListener("click", (ev) => {
    const row = ev.target.closest(".palette-row");
    if (!row) return;
    cursor = parseInt(row.dataset.idx || "0", 10);
    commitCursor();
  });

  /* ── Global hotkey: Ctrl+K / Cmd+K ──────────────────────────── */
  document.addEventListener("keydown", (ev) => {
    const isMac = navigator.platform.toLowerCase().includes("mac");
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

  // Expose for SPA swap to refresh the index if needed
  window.ftsPalette = { open: openPalette, close: closePalette, refresh: () => { navIndex = indexNav(); } };
})();
