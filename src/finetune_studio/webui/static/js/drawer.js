/* ============================================================
   FINETUNE STUDIO — Drawer + breadcrumb popover controller
   ------------------------------------------------------------
   Replaces the persistent left sidebar with:
     1. A slide-in drawer (toggle via [≡] button or '\' key)
     2. A breadcrumb popover on project pages (click project name)
   Both close on backdrop click, Escape, or item selection.
   SPA-aware: state survives navigation.
   ============================================================ */
(function () {
  const DRAWER = "fts.drawer.open";
  const POP = "fts.crumbPop.open";

  function $(id) { return document.getElementById(id); }

  /* ── Drawer ──────────────────────────────────────────────────── */
  const drawer = $("drawer");
  const backdrop = $("drawer-backdrop");
  const toggleBtn = $("nav-toggle");
  const closeBtn = $("drawer-close");

  function openDrawer() {
    if (!drawer || !backdrop) return;
    drawer.hidden = false;
    backdrop.hidden = false;
    // force reflow so the transition runs
    drawer.getBoundingClientRect();
    drawer.classList.add("open");
    backdrop.classList.add("open");
    try { sessionStorage.setItem(DRAWER, "1"); } catch (e) {}
    // Focus first item for keyboard nav
    const first = drawer.querySelector(".drawer-item");
    if (first) first.focus({ preventScroll: true });
  }
  function closeDrawer() {
    if (!drawer || !backdrop) return;
    drawer.classList.remove("open");
    backdrop.classList.remove("open");
    setTimeout(() => {
      drawer.hidden = true;
      backdrop.hidden = true;
    }, 220);
    try { sessionStorage.removeItem(DRAWER); } catch (e) {}
  }
  function toggleDrawer() {
    if (drawer && drawer.hidden) openDrawer(); else closeDrawer();
  }

  if (toggleBtn) toggleBtn.addEventListener("click", toggleDrawer);
  if (closeBtn) closeBtn.addEventListener("click", closeDrawer);
  if (backdrop) backdrop.addEventListener("click", closeDrawer);

  // Auto-close drawer when a nav item is clicked (so the new page renders
  // full-width, not behind the drawer).
  document.addEventListener("click", (ev) => {
    const t = ev.target.closest("[data-drawer-close]");
    if (t) closeDrawer();
  });

  /* ── Breadcrumb popover ──────────────────────────────────────── */
  const popBtn = $("crumb-popover-btn");
  const pop = $("crumb-popover");
  const popClose = $("crumb-pop-close");

  function openPop() {
    if (!pop || !popBtn) return;
    pop.hidden = false;
    pop.getBoundingClientRect();
    pop.classList.add("open");
    popBtn.classList.add("open");
    try { sessionStorage.setItem(POP, "1"); } catch (e) {}
  }
  function closePop() {
    if (!pop || !popBtn) return;
    pop.classList.remove("open");
    popBtn.classList.remove("open");
    setTimeout(() => { pop.hidden = true; }, 180);
    try { sessionStorage.removeItem(POP); } catch (e) {}
  }
  function togglePop() {
    if (!pop) return;
    if (pop.hidden) openPop(); else closePop();
  }

  if (popBtn) popBtn.addEventListener("click", (ev) => {
    ev.stopPropagation();
    togglePop();
  });
  if (popClose) popClose.addEventListener("click", closePop);

  // Click-outside closes popover
  document.addEventListener("click", (ev) => {
    if (!pop || pop.hidden) return;
    if (ev.target.closest("#crumb-popover") || ev.target.closest("#crumb-popover-btn")) return;
    closePop();
  });

  /* ── Keyboard: '\' toggles drawer, Escape closes either ──────── */
  document.addEventListener("keydown", (ev) => {
    // Don't hijack when user is typing
    const t = ev.target;
    if (t && (t.tagName === "INPUT" || t.tagName === "TEXTAREA" || t.isContentEditable)) {
      if (ev.key === "Escape") {
        closeDrawer(); closePop();
      }
      return;
    }
    if (ev.key === "\\" && !ev.ctrlKey && !ev.metaKey && !ev.altKey) {
      ev.preventDefault();
      toggleDrawer();
      return;
    }
    if (ev.key === "Escape") {
      closeDrawer(); closePop();
    }
  });

  /* ── SPA: re-bind after navigation ─────────────────────────────
     Templates are swapped into #content, so per-page scripts may
     re-attach. We only need to keep the drawer references (they
     live in <body>, not in #content). But popover may be replaced
     after a SPA swap if the next page is also a project page, so
     re-lookup. */
  window.ftsDrawer = {
    open: openDrawer,
    close: closeDrawer,
    toggle: toggleDrawer,
    openPop, closePop, togglePop,
  };
})();
